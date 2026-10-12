import argparse
import os
import time

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from dataset import DataSet
from model import PODRec
from trainer import Trainer
from utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)

    # Data
    parser.add_argument('--data_name', type=str, default='jdata', help='choose in {jdata, tmall, taobao}')
    parser.add_argument('--model_name', type=str, default='MBDRM', help='model name')
    parser.add_argument('--model_path', type=str, default='./checkpoint', help='save model path')
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument('--log_path', type=str, default='./log', help='log path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=2023, help='seed')

    # Training & Evaluation
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.0, help='decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=30, help='early stop threshold')
    parser.add_argument('--batch_size', type=int, default=512, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size')
    parser.add_argument('--topk', type=list, default=[10, 20, 50], help='top k')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='metrics')
    parser.add_argument('--training_interval', type=int, default=5, help='training interval')

    # Load & Save Model
    parser.add_argument('--if_load_model', action='store_true', help='load model')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path')

    # params for multi-task learning
    parser.add_argument('--if_multi_tasks', action='store_true', help='multi-task learning')
    parser.add_argument('--aux_weight_loss', type=float, default=1.0, help='loss auxiliary weight')
    parser.add_argument('--aux_weight_score', type=float, default=1.0, help='score auxiliary weight')
    parser.add_argument('--no_impact', type=str, default='no_action', help='ablation auxiliary weight')

    # params for the dnn
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dnn_dims', type=str, default='[8000]', help='the dims for the DNN')
    parser.add_argument('--dnn_norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--timestep_emb_size', type=int, default=10, help='timestep embedding size')

    # params for diffusion
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--diffusion_steps', type=int, default=40, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.001, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0005, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.005, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True,
                        help='assign different weight to different timestep or not')

    args = parser.parse_args()

    # Seed and Device
    set_seed(args.seed)

    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')

    # Log
    args.t = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    full_log_path = os.path.join(args.log_path, args.t)
    if not os.path.exists(full_log_path):
        os.makedirs(full_log_path)
    logger.add(os.path.join(full_log_path, '{}_{}.log'.format(args.model_name, args.data_name)), encoding='utf-8')
    args.train_writer = SummaryWriter(os.path.join(full_log_path, '{}_{}_train'.format(args.model_name, args.data_name)))
    args.valid_writer = SummaryWriter(os.path.join(full_log_path, '{}_{}_validation'.format(args.model_name, args.data_name)))
    args.test_writer = SummaryWriter(os.path.join(full_log_path, '{}_{}_test'.format(args.model_name, args.data_name)))

    # Dataset and Model
    start = time.time()
    dataset = DataSet(args)
    model = PODRec(args, dataset)
    model = model.to(args.device)

    # Train
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    logger.info('total cost time %.2fs' % (time.time() - start))
    logger.info(args.__str__())
