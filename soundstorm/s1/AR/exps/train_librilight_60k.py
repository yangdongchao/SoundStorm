# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from soundstorm.s1.AR.data.data_module_librilight_60k import Text2SemanticDataModule
from soundstorm.s1.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from soundstorm.utils import get_files_by_prefix_suffix
from soundstorm.utils import get_newest_ckpt
from soundstorm.utils.io import load_yaml_config

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
torch.set_float32_matmul_precision('high')


def get_key_name(file_path: str):
    name_list = file_path.split("/")
    # phonemes_0_3.npy -> 0_3
    rank_name = '_'.join(name_list[-1].split('.')[0].split('_')[-2:])
    # small/medium/large/duplicate_0_3
    key_name = f'{name_list[-3]}_{rank_name}'
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


def split_files_by_size(file_list, n):
    # return key_name = 'small_0_3'
    groups = greedy_file_split(file_list, n)
    groups_key_name = [[get_key_name(file_path) for file_path in group]
                       for group in groups]
    return groups, groups_key_name


def get_phoneme_file_groups_by_groups_key_name(phoneme_files, groups_key_name):
    '''
    dict:{key_name: phoneme_file_path}
    '''
    phoneme_key_name_file_dict = dict()
    for phoneme_file in phoneme_files:
        key = get_key_name(phoneme_file)
        phoneme_key_name_file_dict[key] = phoneme_file

    phoneme_file_groups = []
    # groups_key_name [[],[],[]]
    for group_key_name in groups_key_name:
        phoneme_file_group = []
        for key_name in group_key_name:
            phoneme_file_group.append(phoneme_key_name_file_dict[key_name])
        phoneme_file_groups.append(phoneme_file_group)
    return phoneme_file_groups


def get_non_speech_file_groups_by_groups_key_name(non_speech_files,
                                                  groups_key_name):
    '''
    dict:{key_name: non_speech_file_path}
    '''
    non_speech_key_name_file_dict = dict()
    for non_speech_file in non_speech_files:
        key = get_key_name(non_speech_file)
        non_speech_key_name_file_dict[key] = non_speech_file

    non_speech_file_groups = []
    # groups_key_name [[],[],[]]
    for group_key_name in groups_key_name:
        non_speech_file_group = []
        for key_name in group_key_name:
            if key_name in non_speech_key_name_file_dict:
                non_speech_file_group.append(
                    non_speech_key_name_file_dict[key_name])
        non_speech_file_groups.append(non_speech_file_group)
    return non_speech_file_groups


def check_shapes(list1, list2):
    # 检查子列表数量是否相等
    if len(list1) != len(list2):
        return False

    # 检查每个子列表的长度是否相等
    for sublist1, sublist2 in zip(list1, list2):
        if len(sublist1) != len(sublist2):
            return False
    return True


# semantic_dirs, phoneme_dirs, 总 rank 数，当前 rank
# 要保证每个 rank 分到的 split data 不重不漏，也就是只 random 一次
# dist.get_rank()
# 最后返回一个 list(list)，list 的长度是 global_rank_num 的长度
def get_datasplit_for_rank(semantic_dirs,
                           phoneme_dirs,
                           global_rank_num: int,
                           non_speech_dirs=None):
    all_semantic_files = []
    all_phoneme_files = []
    all_non_speech_files = []

    for semantic_dir in semantic_dirs:
        all_semantic_files += get_files_by_prefix_suffix(
            semantic_dir, prefix='semantic_token', suffix='tsv')
    for phoneme_dir in phoneme_dirs:
        all_phoneme_files += get_files_by_prefix_suffix(
            phoneme_dir, prefix='phonemes', suffix='npy')

    if non_speech_dirs is not None:
        for non_speech_dir in non_speech_dirs:
            all_non_speech_files += get_files_by_prefix_suffix(
                non_speech_dir, prefix='non_speech', suffix='npy')

    # [[file_path1, file_path2],[file_path3, file_path4],[file_path5, file_path6]]
    semantic_file_groups, groups_key_name = split_files_by_size(
        all_semantic_files, n=global_rank_num)
    # phoneme 一定要和 semantic 对应才行
    # 按照 groups_key_name 分配 all_phoneme_files
    phoneme_file_groups = get_phoneme_file_groups_by_groups_key_name(
        all_phoneme_files, groups_key_name)

    non_speech_file_groups = get_non_speech_file_groups_by_groups_key_name(
        all_non_speech_files, groups_key_name)

    semantic_file_groups_sizes = [
        sum(os.path.getsize(file) for file in group)
        for group in semantic_file_groups
    ]
    phoneme_file_groups_sizes = [
        sum(os.path.getsize(file) for file in group)
        for group in phoneme_file_groups
    ]
    print("semantic_file_groups_sizes:", semantic_file_groups_sizes)
    print("phoneme_file_groups_sizes:", phoneme_file_groups_sizes)
    assert check_shapes(semantic_file_groups, phoneme_file_groups) is True
    return semantic_file_groups, phoneme_file_groups, non_speech_file_groups


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(args.config_file)

    seed_everything(config["train"]["seed"], workers=True)

    ckpt_callback: ModelCheckpoint = ModelCheckpoint(
        save_top_k=-1,
        save_on_train_epoch_end=False,
        every_n_train_steps=config["train"]["every_n_train_steps"],
        dirpath=ckpt_dir)
    logger = WandbLogger(
        project="AR_S1_LibriLight",
        name=output_dir.stem,
        save_dir=output_dir,
        # resume the loss curve
        resume=True,
        # id='k19kvsq8'
    )
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator='gpu',
        devices=-1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=config["train"]["precision"],
        logger=logger,
        callbacks=[ckpt_callback])

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        config, output_dir)

    global_rank = trainer.global_rank
    print(f"当前进程的 global_rank 是: {global_rank}")
    local_rank = trainer.local_rank
    print(f"当前进程的 local_rank 是: {local_rank}")
    world_size = trainer.world_size
    print(f"当前进程的 world_size 是: {world_size}")

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_paths=args.train_semantic_file_groups[global_rank],
        train_phoneme_paths=args.train_phoneme_file_groups[global_rank],
        dev_semantic_paths=args.dev_semantic_file_groups[global_rank],
        dev_phoneme_paths=args.dev_phoneme_file_groups[global_rank],
        train_non_speech_paths=args.train_non_speech_file_groups[global_rank],
        dev_non_speech_paths=args.dev_non_speech_file_groups[global_rank],
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size)
    try:
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None

    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/default.yaml',
        help='path of config file')
    # args for dataset
    parser.add_argument(
        '--train_semantic_dirs',
        type=str,
        nargs='*',
        default="dump/small/train/",
        help='dirs of train semantic')
    parser.add_argument(
        '--train_phoneme_dirs',
        type=str,
        nargs='*',
        default="dump/small/train/",
        help='dirs of train phoneme')
    parser.add_argument(
        '--dev_semantic_dirs',
        type=str,
        nargs='*',
        default="dump/small/dev/",
        help='dirs of dev semantic')
    parser.add_argument(
        '--dev_phoneme_dirs',
        type=str,
        nargs='*',
        default="dump/small/dev/",
        help='dirs of dev phoneme')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='exp/default',
        help='directory to save the results')

    parser.add_argument(
        '--train_non_speech_dirs',
        type=str,
        nargs='*',
        default=None,
        help='dirs of train non_speech data')

    parser.add_argument(
        '--dev_non_speech_dirs',
        type=str,
        nargs='*',
        default=None,
        help='dirs of dev non_speech data')

    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    num_node = 1
    world_size = ngpus_per_node * num_node

    print(f"当前进程组中的总进程数：{world_size}")

    train_semantic_file_groups, train_phoneme_file_groups, train_non_speech_file_groups = get_datasplit_for_rank(
        semantic_dirs=args.train_semantic_dirs,
        phoneme_dirs=args.train_phoneme_dirs,
        non_speech_dirs=args.train_non_speech_dirs,
        global_rank_num=world_size)

    args.train_semantic_file_groups = train_semantic_file_groups
    args.train_phoneme_file_groups = train_phoneme_file_groups
    args.train_non_speech_file_groups = train_non_speech_file_groups

    dev_semantic_file_groups, dev_phoneme_file_groups, dev_non_speech_file_groups = get_datasplit_for_rank(
        semantic_dirs=args.dev_semantic_dirs,
        phoneme_dirs=args.dev_phoneme_dirs,
        non_speech_dirs=args.dev_non_speech_dirs,
        global_rank_num=world_size)

    args.dev_semantic_file_groups = dev_semantic_file_groups
    args.dev_phoneme_file_groups = dev_phoneme_file_groups
    args.dev_non_speech_file_groups = dev_non_speech_file_groups

    print("args.train_semantic_file_groups:", args.train_semantic_file_groups)
    print("args.train_phoneme_file_groups:", args.train_phoneme_file_groups)
    print("args.train_non_speech_file_groups:",
          args.train_non_speech_file_groups)

    logging.info(str(args))

    main(args)
