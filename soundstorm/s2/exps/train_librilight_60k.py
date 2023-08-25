# 这里需要新增一个 main3_ddp.py 里面的 get_content 函数，为每个 rank 分配几个 split
# train and eval control by iter not epoch
# ------------------------------------------
# Diffsound, By Dongchao Yang
# based on https://github.com/cientgu/VQ-Diffusion
# ------------------------------------------
import argparse
import os
import time
import warnings
from academicodec.models.hificodec.vqvae import VQVAE

import torch
from soundstorm.s2.data.build_librilight_60k import build_dataloader
from soundstorm.s2.distributed.launch import launch
from soundstorm.s2.engine.logger import Logger
from soundstorm.s2.engine.solver_60k import Solver
from soundstorm.s2.models.dalle_wav.build import build_model
from soundstorm.s2.utils.misc import merge_opts_to_config
from soundstorm.s2.utils.misc import modify_config_for_debug
from soundstorm.s2.utils.misc import seed_everything
from soundstorm.utils import get_files_by_prefix_suffix
from soundstorm.utils import str2bool
from soundstorm.utils.io import load_yaml_config

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
        help='path to model that need to be loaded, used for loading pretrained model'
    )
    parser.add_argument(
        "--auto_resume",
        type=str2bool,
        default=True,
        help="automatically resume the training")
    # args for dataset
    parser.add_argument(
        '--train_semantic_dirs',
        type=list,
        nargs='+',
        default=["dump/small/train/"],
        help='dirs of train semantic')
    parser.add_argument(
        '--train_acoustic_dirs',
        type=list,
        nargs='+',
        default=["dump/small/train/acoustic/"],
        help='dirs of train acoustic')
    parser.add_argument(
        '--dev_semantic_dirs',
        type=list,
        nargs='+',
        default=["dump/small/dev/"],
        help='dirs of dev semantic')
    parser.add_argument(
        '--dev_acoustic_dirs',
        type=list,
        nargs='+',
        default=["dump/small/dev/acoustic/"],
        help='dirs of dev acoustic')

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
    # for HiFi-Codec
    parser.add_argument(
        "--hificodec_model_path",
        type=str,
        default='pretrained_model/hificodec//HiFi-Codec-16k-320d')
    parser.add_argument(
        "--hificodec_config_path",
        type=str,
        default='pretrained_model/hificodec/config_16k_320d.json')

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER, )

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    new_train_semantic_dirs = []
    new_train_acoustic_dirs = []
    new_dev_semantic_dirs = []
    new_dev_acoustic_dirs = []
    # format dataset dirs
    for item in args.train_semantic_dirs:
        new_train_semantic_dirs.append(''.join(item))
    args.train_semantic_dirs = new_train_semantic_dirs

    for item in args.train_acoustic_dirs:
        new_train_acoustic_dirs.append(''.join(item))
    args.train_acoustic_dirs = new_train_acoustic_dirs

    for item in args.dev_semantic_dirs:
        new_dev_semantic_dirs.append(''.join(item))
    args.dev_semantic_dirs = new_dev_semantic_dirs

    for item in args.dev_acoustic_dirs:
        new_dev_acoustic_dirs.append(''.join(item))
    args.dev_acoustic_dirs = new_dev_acoustic_dirs

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
        args.world_size = args.ngpus_per_node * args.num_node
    # 8 
    # train 和 dev 不一定相同 split 的就要分到一起？
    # 那 dev 的时候只在 0 卡 dev 的话，见到的数据永远是 0 卡分到的数据？
    # ❗️ 看下是不是 SequentialSampler 影响的
    # [[],[],[],[]]
    train_semantic_file_groups, train_acoustic_file_groups = get_datasplit_for_rank(
        semantic_dirs=args.train_semantic_dirs,
        acoustic_dirs=args.train_acoustic_dirs,
        global_rank_num=args.world_size)

    args.train_semantic_file_groups = train_semantic_file_groups
    args.train_acoustic_file_groups = train_acoustic_file_groups

    dev_semantic_file_groups, dev_acoustic_file_groups = get_datasplit_for_rank(
        semantic_dirs=args.dev_semantic_dirs,
        acoustic_dirs=args.dev_acoustic_dirs,
        global_rank_num=args.world_size)

    args.dev_semantic_file_groups = dev_semantic_file_groups
    args.dev_acoustic_file_groups = dev_acoustic_file_groups

    launch(
        main_worker,
        args.ngpus_per_node,
        args.num_node,
        args.node_rank,
        args.dist_url,
        args=(args, ))


def get_semantic_file_key_name(semantic_file):
    name_list = semantic_file.split("/")
    # semantic_token_0_3.tsv -> 0_3
    rank_name = '_'.join(name_list[-1].split('.')[0].split('_')[-2:])
    # small/medium/large/duplicate_0_3
    key_name = f'{name_list[-3]}_{rank_name}'
    return key_name


def get_acoustic_file_key_name(acoustic_file):
    name_list = acoustic_file.split("/")
    rank_name = '_'.join(name_list[-1].split('.')[0].split('_')[-2:])
    key_name = f'{name_list[-4]}_{rank_name}'
    return key_name


# 用非空子列表填充空子列表
# 可能会导致某些 data split 有多张卡都在用
# 如有 7 个 split 但是有 8 张卡，则 split 0 会用 2 遍
def fill_empty_group(original_list):
    # 找出非空子列表
    non_empty_sublists = [sublist for sublist in original_list if sublist]
    result_list = []
    for sublist in original_list:
        if sublist:
            result_list.append(sublist)
        else:
            non_empty_index = len(result_list) % len(non_empty_sublists)
            result_list.append(non_empty_sublists[non_empty_index])
    return result_list


# 此处需要保证 len(file_list) >= n
def greedy_file_split(file_list, n):
    # 按文件大小从大到小排序
    file_list.sort(key=os.path.getsize, reverse=True)
    total_size = sum(os.path.getsize(file) for file in file_list)
    avg_size_per_group = total_size // n
    groups = [[] for _ in range(n)]
    group_sizes = [0] * n

    for file in file_list:
        smallest_group = min(range(n), key=lambda i: group_sizes[i])
        groups[smallest_group].append(file)
        group_sizes[smallest_group] += os.path.getsize(file)

    # len(file_list) < n 时, 用非空子列表填充空子列表
    groups = fill_empty_group(groups)
    return groups


# 当 len(file_list) <  n 时，会有空的 group


def split_files_by_size(file_list, n):
    # return key_name = 'small_0_3'
    groups = greedy_file_split(file_list, n)
    groups_key_name = [
        [get_semantic_file_key_name(file_path) for file_path in group]
        for group in groups
    ]
    return groups, groups_key_name


def get_acoustic_file_groups_by_groups_key_name(acoustic_files,
                                                groups_key_name):
    '''
    dict:{key_name: acoustic_file_path}
    '''
    acoustic_key_name_file_dict = dict()
    for acoustic_file in acoustic_files:
        key = get_acoustic_file_key_name(acoustic_file)
        acoustic_key_name_file_dict[key] = acoustic_file

    acoustic_file_groups = []
    # groups_key_name [[],[],[]]
    for group_key_name in groups_key_name:
        acoustic_file_group = []
        for key_name in group_key_name:
            acoustic_file_group.append(acoustic_key_name_file_dict[key_name])
        acoustic_file_groups.append(acoustic_file_group)
    return acoustic_file_groups


def check_shapes(list1, list2):
    # 检查子列表数量是否相等
    if len(list1) != len(list2):
        return False

    # 检查每个子列表的长度是否相等
    for sublist1, sublist2 in zip(list1, list2):
        if len(sublist1) != len(sublist2):
            return False
    return True


# semantic_dirs, acoustic_dirs, 总 rank 数，当前 rank
# 要保证每个 rank 分到的 split data 不重不漏，也就是只 random 一次
# dist.get_rank()
# 最后返回一个 list(list)，list 的长度是 global_rank_num 的长度
def get_datasplit_for_rank(semantic_dirs, acoustic_dirs, global_rank_num: int):
    all_semantic_files = []
    all_acoustic_files = []
    for semantic_dir in semantic_dirs:
        all_semantic_files += get_files_by_prefix_suffix(
            semantic_dir, prefix='semantic_token', suffix='tsv')
    for acoustic_dir in acoustic_dirs:
        all_acoustic_files += get_files_by_prefix_suffix(
            acoustic_dir, prefix='hificodec', suffix='pth')
    # [[file_path1, file_path2],[file_path3, file_path4],[file_path5, file_path6]]
    semantic_file_groups, groups_key_name = split_files_by_size(
        all_semantic_files, n=global_rank_num)
    # acoustic 一定要和 semantic 对应才行
    # 按照 groups_key_name 分配 all_acoustic_files
    acoustic_file_groups = get_acoustic_file_groups_by_groups_key_name(
        all_acoustic_files, groups_key_name)

    semantic_file_groups_sizes = [
        sum(os.path.getsize(file) for file in group)
        for group in semantic_file_groups
    ]
    acoustic_file_groups_sizes = [
        sum(os.path.getsize(file) for file in group)
        for group in acoustic_file_groups
    ]
    print("semantic_file_groups_sizes:", semantic_file_groups_sizes)
    print("acoustic_file_groups_sizes:", acoustic_file_groups_sizes)
    assert check_shapes(semantic_file_groups, acoustic_file_groups) is True
    return semantic_file_groups, acoustic_file_groups


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1
    # load config
    config = load_yaml_config(args.config_file)
    # 合并命令行输入到 config 文件中
    config = merge_opts_to_config(config, args.opts)
    if args.debug:
        config = modify_config_for_debug(config)
    # get logger
    logger = Logger(args)
    logger.save_config(config)
    '''
    # get model 
    model = build_model(config)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # for sample()
    # NOTE by yuantian: all threads use some of memory of GPU 0 which need to be fixed
    if local_rank == 0:
        hificodec = VQVAE(
            config_path=args.hificodec_config_path,
            ckpt_path=args.hificodec_model_path,
            with_encoder=True)
        hificodec.generator.remove_weight_norm()
        hificodec.encoder.remove_weight_norm()
        hificodec.eval()
    else:
        hificodec = None
    '''
    # get dataloader
    print("start build dataloader...")
    start_build_time = time.time()
    dataloader_info = build_dataloader(config, args)
    print(f"time of build dataloader: {time.time() - start_build_time}")
    '''
    # 每个 rank 都有自己的 dataloader, 每个 dataloader 加载不同的 split
    # solver_60k must train with iter
    solver = Solver(
        config=config,
        args=args,
        model=model,
        dataloader=dataloader_info,
        logger=logger,
        hificodec=hificodec)

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
    torch.cuda.empty_cache()
    '''


if __name__ == '__main__':
    main()
