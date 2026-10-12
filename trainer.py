import os
import time

import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DataSet
from metrics import metrics_dict


class Trainer(object):

    def __init__(self, model, dataset: DataSet, args):
        self.model = model.to(args.device)
        self.dataset = dataset
        self.behaviors = dataset.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.train_writer = args.train_writer
        self.test_writer = args.test_writer
        self.device = args.device
        self.t = args.t
        self.early_stop = args.early_stop
        self.training_interval = args.training_interval

        self.optimizer = self.get_optimizer(self.model)

    def get_optimizer(self, model):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer

    @logger.catch()
    def train_model(self):
        train_dataset_loader = DataLoader(dataset=self.dataset.behavior_dataset(),
                                          batch_size=self.batch_size,
                                          shuffle=True)

        best_valid_result = 0
        best_valid_epoch = 0

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self._train_one_epoch(train_dataset_loader, epoch)

            if epoch % self.training_interval == 0:
                # test
                start_time = time.time()

                validate_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.test_dataset(),
                                                 self.dataset.validation_dict, self.dataset.validation_gt_length,
                                                 self.test_writer)
                epoch_time = time.time() - start_time
                logger.info(f"validate %d, time %.2fs, result: %s " % (epoch, epoch_time, validate_metric_dict.__str__()))

                valid_result = validate_metric_dict['ndcg@10']
                # update
                if valid_result - best_valid_result > 0:
                    best_valid_result = valid_result
                    best_valid_epoch = epoch
                    self.save_model(self.model)
                    logger.info(f"model saved at epoch %d" % epoch)

            # early stop
            if epoch - best_valid_epoch >= self.early_stop or epoch == self.epochs:
                # 加载验证集最优的模型
                full_model_path = os.path.join(self.model_path, self.t)
                self.model.load_state_dict(torch.load(os.path.join(full_model_path, 'model.pth')))
                self.model.storage_all_embeddings = torch.load(
                    os.path.join(full_model_path, 'all_embeddings.pth')
                )
                logger.info(f"loaded best model from epoch {best_valid_epoch}")

                start_time = time.time()
                test_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.test_dataset(),
                                                 self.dataset.test_dict, self.dataset.test_gt_length,
                                                 self.test_writer)
                epoch_time = time.time() - start_time
                logger.info(f"test %d, time %.2fs, result: %s " % (best_valid_epoch, epoch_time, test_metric_dict.__str__()))
                break


    def _train_one_epoch(self, behavior_dataset_loader, epoch):
        start_time = time.time()
        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=len(behavior_dataset_loader),
                desc=f"\033[1;35m Train {epoch:>5}\033[0m"
            )
        )

        # train
        total_loss = 0.0
        batch_no = 0
        for batch_index, (batch_data, users) in behavior_dataset_iter:
            # users = users.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(users)
            loss.backward()
            self.optimizer.step()
            batch_no = batch_index + 1
            total_loss += loss.item()
        total_loss = total_loss / batch_no

        self.train_writer.add_scalar('total train loss', total_loss, epoch)
        epoch_time = time.time() - start_time
        logger.info('epoch %d, time %.2fs, train loss: %.4f ' % (epoch, epoch_time, total_loss))


    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length, writer):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)

        self.model.eval()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.dataset.train_user_behavior_dict[self.behaviors[-1]]
        for batch_index, (batch_data, users) in iter_data:
            # users = users.to(self.device)
            scores = self.model.full_predict(users)  # (test_bsz, |I|)

            users = users.to('cpu')
            scores = scores.to('cpu')
            for index, user in enumerate(users):

                if user.item() not in gt_interacts:
                    continue

                user_score = scores[index]
                items = train_items.get(user.item(), None)
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                user = user
                gt_items = gt_interacts[user.item()]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        for key, value in metric_dict.items():
            writer.add_scalar('evaluate ' + key, value, epoch)
        return metric_dict

    @logger.catch()
    @torch.no_grad()
    def final_evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length, writer, aux_weight):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)

        self.model.eval()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.dataset.train_user_behavior_dict[self.behaviors[-1]]
        for batch_index, (batch_data, users) in iter_data:
            # users = users.to(self.device)
            scores = self.model.full_predict(users, aux_weight)  # (test_bsz, |I|)

            users = users.to('cpu')
            scores = scores.to('cpu')
            for index, user in enumerate(users):

                if user.item() not in gt_interacts:
                    continue

                user_score = scores[index]
                items = train_items.get(user.item(), None)
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                user = user
                gt_items = gt_interacts[user.item()]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        for key, value in metric_dict.items():
            writer.add_scalar('evaluate ' + key, value, epoch)
        return metric_dict

    def calculate_result(self, topk_list, gt_len):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = '{}@{}'.format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, model):
        full_model_path = os.path.join(self.model_path, self.t)
        if not os.path.exists(full_model_path):
            os.makedirs(full_model_path)
        torch.save(model.state_dict(), os.path.join(full_model_path, 'model.pth'))
        torch.save(model.storage_all_embeddings, os.path.join(full_model_path, 'all_embeddings.pth'))

        return os.path.join(full_model_path, 'model.pth')
