# ------------------------------------------
# Diffsound, By Dongchao Yang
# based on https://github.com/cientgu/VQ-Diffusion
# ------------------------------------------
import argparse
import datetime
import os
import time
import warnings

import numpy as np
import torch
from soundstorm.s2.data.build import build_dataloader
from soundstorm.s2.distributed.launch import launch
from soundstorm.s2.engine.logger import Logger
from soundstorm.s2.engine.solver import Solver
from soundstorm.s2.models.dalle_wav.build import build_model
from soundstorm.s2.utils.io import load_yaml_config
from soundstorm.s2.utils.misc import merge_opts_to_config
from soundstorm.s2.utils.misc import modify_config_for_debug
from soundstorm.s2.utils.misc import seed_everything
from soundstorm.utils import str2bool

NODE_RANK = os.environ['INDEX'] if 'INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = (os.environ['CHIEF_IP'],
                            22275) if 'CHIEF_IP' in os.environ else (
                                "127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
NUM_NODE = os.environ['HOST_NUM'] if 'HOST_NUM' in os.environ else 1


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training script')
    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/default.yaml',
        help='path of config file')
    parser.add_argument(
        '--output',
        type=str,
        default='exp/default',
        help='directory to save the results')
    parser.add_argument(
        '--log_frequency',
        type=int,
        default=100,
        help='print frequency (default: 100 iter)')
    parser.add_argument(
        '--load_path',
        type=str,
        default=None,
        help='path to model that need to be loaded, used for loading pretrained model')
    parser.add_argument(
        "--auto_resume",
        type=str2bool,
        default=True,
        help="automatically resume the training")
    # args for dataset
    parser.add_argument(
        '--train_semantic_path',
        type=str,
        default='dump/train/semantic_token.tsv')
    parser.add_argument(
        '--train_acoustic_path',
        type=str,
        default='dump/train/acoustic_token/hificodec.pth')
    parser.add_argument(
        '--dev_semantic_path', type=str, default='dump/dev/semantic_token.tsv')
    parser.add_argument(
        '--dev_acoustic_path',
        type=str,
        default='dump/dev/acoustic_token/hificodec.pth')

    # args for ddp
    parser.add_argument(
        '--num_node',
        type=int,
        default=NUM_NODE,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--ngpus_per_node',
        type=int,
        default=8,
        help='number of gpu on one node')
    parser.add_argument(
        '--node_rank',
        type=int,
        default=NODE_RANK,
        help='node rank for distributed training')
    parser.add_argument(
        '--dist_url',
        type=str,
        default=DIST_URL,
        help='url used to set up distributed training')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU id to use. If given, only the specific gpu will be'
        ' used, and ddp will be disabled')
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    parser.add_argument(
        "--sync_bn", type=str2bool, default=False, help="use sync BN layer")
    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="use tensorboard for logging")
    parser.add_argument("--timestamp", type=str2bool, default=True)
    # args for random
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='seed for initializing training. ')
    parser.add_argument(
        "--cudnn_deterministic",
        type=str2bool,
        default=False,
        help="set cudnn.deterministic True")
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=False,
        help="automatic mixture of precesion")
    parser.add_argument(
        "--debug", type=str2bool, default=False, help="set as debug mode")

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER, )
    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))
    # modify args for debugging
    if args.debug:
        if args.gpu is None:
            args.gpu = 0
    return args


def main():
    args = get_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
    if args.gpu is not None:
        warnings.warn(
            'You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        print('args.num_node ', args.num_node)
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node  # 
    launch(
        main_worker,
        args.ngpus_per_node,
        args.num_node,
        args.node_rank,
        args.dist_url,
        args=(args, ))


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1
    print(args)
    # load config
    config = load_yaml_config(args.config_file)
    # 合并命令行输入到 config 文件中
    config = merge_opts_to_config(config, args.opts)
    if args.debug:
        config = modify_config_for_debug(config)
    # get logger
    logger = Logger(args)
    logger.save_config(config)

    # get model 
    model = build_model(config, args)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # get dataloader
    dataloader_info = build_dataloader(config, args)
    # get solver
    solver = Solver(
        config=config,
        args=args,
        model=model,
        dataloader=dataloader_info,
        logger=logger)

    # resume 
    # only load the model paramters
    if args.load_path is not None:
        solver.resume(
            path=args.load_path,
            # load_model=True,
            load_optimizer_and_scheduler=False,
            load_others=False)
    elif args.auto_resume:
        print("in auto_resume")
        solver.resume()
    solver.train()


if __name__ == '__main__':
    main()
