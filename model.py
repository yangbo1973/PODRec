import os.path
from collections import defaultdict
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import DataSet
from dnn import DNN
import gaussian_diffusion as gd


def get_subsets_by_max(input_list):
    # all non-empty subsets of behaviors
    subsets = []
    n = len(input_list)
    for r in range(1, n + 1):  # non-empty
        subsets.extend(combinations(input_list, r))

    # dictionary：key=local target beh，value=list of subset
    max_dict = {}
    for subset in subsets:
        max_val = max(subset)
        if max_val not in max_dict:
            max_dict[max_val] = []
        max_dict[max_val].append(list(subset))  # turn to list

    return max_dict

class PODRec(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(PODRec, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.checkpoint = args.checkpoint
        self.if_load_model = args.if_load_model
        self.data_name = args.data_name

        # Base
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count

        # Modules
        self.behaviors = dataset.behaviors
        self.behavior_adjs = dataset.behavior_mats
        self.behavior_index_dict = get_subsets_by_max(list(range(len(self.behaviors))))
        for beh in self.behaviors:
            self.behavior_adjs[beh] = self.behavior_adjs[beh].to_dense()
        self.behavior_trans_adjs = []

        self.behavior_cf = nn.ModuleDict()
        out_dims = eval(args.dnn_dims) + [self.n_items]
        in_dims = out_dims[::-1]
        for post_beh in self.behaviors:
            self.behavior_cf[post_beh] = DNN(in_dims, out_dims, args.timestep_emb_size, time_type="cat", norm=args.dnn_norm).to(self.device)

        self.diffusion = gd.GaussianDiffusion(args.mean_type, args.noise_schedule,
                                         args.noise_scale, args.noise_min, args.noise_max, args.diffusion_steps, self.device).to(self.device)

        # Loss
        self.aux_weight_score = args.aux_weight_score
        self.aux_weight_loss = args.aux_weight_loss
        self.no_impact = args.no_impact
        self.if_multi_tasks = args.if_multi_tasks

        self.storage_all_embeddings = None

        # self._init_weights()
        self._load_model()

        self.reweight = args.reweight
        self.sampling_steps = args.sampling_steps
        self.sampling_noise = args.sampling_noise

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.checkpoint, 'model.pth'))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self, users):

        self.behavior_trans_adjs = []
        for index, beh in enumerate(self.behaviors):
            sub_behavior_trans_adjs = []
            for sub_list in self.behavior_index_dict[index]:
                behavior_trans_adjs = torch.zeros_like(self.behavior_adjs[self.behaviors[-1]][users])
                for sl_index in sub_list:
                    behavior_trans_adjs = (behavior_trans_adjs.bool() | self.behavior_adjs[self.behaviors[sl_index]][users].bool()).float()
                sub_behavior_trans_adjs.append(behavior_trans_adjs)
            self.behavior_trans_adjs.append(sub_behavior_trans_adjs)

        terms = {}
        for index, post_beh in enumerate(self.behaviors):
            if post_beh == self.no_impact:
                continue
            else:
                model = self.behavior_cf[post_beh]
                term = {}
                term["loss"] = 0.0
                for i, sub_list in enumerate(self.behavior_index_dict[index]):
                    x_start = self.behavior_trans_adjs[index][i].to(self.device)
                    term["loss"] += self.diffusion.training_losses(model, x_start, self.reweight, target=self.behavior_adjs[self.behaviors[index]][users].to(self.device))["loss"]
                terms[post_beh] = term
        return terms

    def forward(self, batch_data):
        self.storage_all_embeddings = None
        total_loss = 0.0
        terms = self.gcn_propagate(batch_data)  # dict (|B|, N, dim)
        for index, behavior in enumerate(self.behaviors):
            if self.if_multi_tasks or behavior == self.behaviors[-1]:
                if behavior == self.no_impact:
                    continue
                else:
                    if behavior == self.behaviors[-1]:
                        term = terms[behavior]["loss"]
                    else:
                        # users = users.to(self.personal_loss_ratios[behavior].device)  #新插入
                        term = self.aux_weight_loss * terms[behavior]["loss"]
                    # user_loss_ratios = user_loss_ratios[mask]
                    total_loss += term.mean()

        return total_loss

    def full_predict(self, users, aux_weight=-1.):

        self.behavior_trans_adjs = []
        for index, beh in enumerate(self.behaviors):
            sub_behavior_trans_adjs = []
            for sub_list in self.behavior_index_dict[index]:
                behavior_trans_adjs = torch.zeros_like(self.behavior_adjs[self.behaviors[-1]][users])
                for sl_index in sub_list:
                    behavior_trans_adjs = (behavior_trans_adjs.bool() | self.behavior_adjs[self.behaviors[sl_index]][users].bool()).float()
                sub_behavior_trans_adjs.append(behavior_trans_adjs)
            self.behavior_trans_adjs.append(sub_behavior_trans_adjs)

        x_predict = 0.0
        for index, post_beh in enumerate(self.behaviors):
            if post_beh == self.no_impact:
                continue
            else:
                model = self.behavior_cf[post_beh]

                if post_beh == self.behaviors[-1]:
                    for i, sub_list in enumerate(self.behavior_index_dict[index]):
                        x_start = self.behavior_trans_adjs[index][i].to(self.device)
                        x_predict += self.diffusion.p_sample(model, x_start, self.sampling_steps, self.sampling_noise)
                else:
                    for i, sub_list in enumerate(self.behavior_index_dict[index]):
                        x_start = self.behavior_trans_adjs[index][i].to(self.device)
                        if aux_weight < 0:
                            x_predict += self.aux_weight_score * self.diffusion.p_sample(model, x_start, self.sampling_steps, self.sampling_noise)
                        else:
                            x_predict += aux_weight * self.diffusion.p_sample(model, x_start,
                                                                                         self.sampling_steps,
                                                                                         self.sampling_noise)

        self.storage_all_embeddings = x_predict
        scores = x_predict
        return scores
