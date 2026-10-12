import json
import os
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from loguru import logger
from torch.utils.data import Dataset

from utils import sp_mat_to_torch_sp_tensor, normalize_adj, dict2set

def items_to_onehot(items, n_items):

    onehot = torch.zeros(n_items, dtype=torch.float32)

    if not items:
        return onehot
    items_tensor = torch.tensor(items, dtype=torch.long)

    onehot.scatter_(0, items_tensor, 1.0)

    return onehot

class MBTrainData(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

    def __getitem__(self, idx):
        total = []
        for behavior in self.behaviors:

            items = self.behavior_dict[behavior].get(idx, None)
            if items is None:
                signal = torch.zeros(self.item_count)
            else:
                signal = items_to_onehot(items, self.item_count)
            total.append(signal)
        return torch.stack(total), idx

    def __len__(self):
        return self.user_count

class DataSet(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_name = args.data_name
        self.train_user_behavior_dict = defaultdict(dict)
        self.train_item_behavior_dict = defaultdict(dict)

        self.behavior_adjs = {}
        self.behavior_mats = {}

        self.get_behaviors()
        self.get_count()
        self.get_behavior_items()

        self.get_validation_dict()
        self.get_test_dict()
        self.get_interaction_mats()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_dict.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_dict.items()])

    def get_behaviors(self):
        data2behaviors = {
            'taobao': ['click', 'cart', 'buy'],
            'tmall':['click', 'collect', 'cart', 'buy'],
        }

        self.behaviors = data2behaviors[self.data_name]

    def get_count(self):
        with open(os.path.join(self.data_path, self.data_name, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']
            self.users = np.arange(0, self.user_count).tolist()
            self.items = np.arange(0, self.item_count).tolist()
            logger.info('Number of users: {}.'.format(self.user_count))
            logger.info('Number of items: {}.'.format(self.item_count))

    def get_inter_num(self, b_dict):
        return np.sum([len(v) for k, v in b_dict.items()])

    def get_behavior_items(self):
        for behavior in self.behaviors:
            with open(os.path.join(self.data_path, self.data_name, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                b_dict = {int(k): v for k, v in b_dict.items()}
                logger.info('Number of {} interactions: {}'.format(behavior, self.get_inter_num(b_dict)))
                self.train_user_behavior_dict[behavior] = b_dict

        all_dict = defaultdict(list)
        for behavior, b_dict in self.train_user_behavior_dict.items():
            for k, v in b_dict.items():
                all_dict[k].extend(v)
        for k, v in all_dict.items():
            items = list(set(v))
            all_dict[k] = sorted(items)
        self.train_user_behavior_dict['all'] = all_dict

    def get_test_dict(self):
        with open(os.path.join(self.data_path, self.data_name, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            b_dict = {int(k): v for k, v in b_dict.items()}
            self.test_dict = b_dict
            logger.info('Number of test interactions: {}'.format(self.get_inter_num(self.test_dict)))

    def get_validation_dict(self):
        with open(os.path.join(self.data_path, self.data_name, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            b_dict = {int(k): v for k, v in b_dict.items()}
            self.validation_dict = b_dict
            logger.info('Number of validation interactions: {}'.format(self.get_inter_num(self.validation_dict)))

    def get_interaction_mats(self):
        for behavior in self.behaviors:
            row = []
            col = []
            val = []
            for user, items in self.train_user_behavior_dict[behavior].items():
                for item in items:
                    row.append(int(user))
                    col.append(int(item))
                    val.append(1.0)
            mat = sp.coo_matrix((val, (row, col)), shape=(self.user_count, self.item_count))
            a = sp.csr_matrix((self.user_count, self.user_count))
            b = sp.csr_matrix((self.item_count, self.item_count))
            adj = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
            self.behavior_mats[behavior] = sp_mat_to_torch_sp_tensor(mat)


    def behavior_dataset(self):
        return MBTrainData(self.user_count, self.item_count, self.train_user_behavior_dict, self.behaviors)

    def validate_dataset(self):
        return MBTrainData(self.user_count, self.item_count, self.train_user_behavior_dict, self.behaviors)

    def test_dataset(self):
        return MBTrainData(self.user_count, self.item_count, self.train_user_behavior_dict, self.behaviors)
