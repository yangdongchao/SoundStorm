# Fast loader
# it can help to fast read data, so that it can improve the training time.
import random

import torch
import torch.distributed as dist
from soundstorm.s2.data.semantic_dataset_librilight_60k import DDPSyncSampler
from soundstorm.s2.data.semantic_dataset_librilight_60k import SequentialSampler
from soundstorm.s2.utils.misc import instantiate_from_config
from torch.utils.data import ConcatDataset


def build_dataloader(config, args=None, return_dataset=False):
    seed = 999
    # Make sure this is identical to each GPU
    random.seed(seed)

    # global rank
    rank = dist.get_rank()
    print("args.global_rank:", args.global_rank)
    # 两种计算方式结果应该一致
    assert args.global_rank == rank
    # print("args.train_semantic_file_groups[args.global_rank]:",
    #       args.train_semantic_file_groups[args.global_rank])
    # print("-----------------------------------------------------------")
    # print("args.dev_semantic_file_groups[args.global_rank]:",
    #       args.dev_semantic_file_groups[args.global_rank])

    dataset_cfg = config['dataloader']
    batch_size = 1
    train_dataset = []
    for ds_cfg in dataset_cfg['train_datasets']:
        ds_cfg['params']['semantic_paths'] = args.train_semantic_file_groups[
            args.global_rank]
        ds_cfg['params']['acoustic_paths'] = args.train_acoustic_file_groups[
            args.global_rank]
        ds_cfg['params']['max_token_one_batch'] = dataset_cfg[
            'max_token_one_batch']
        ds = instantiate_from_config(ds_cfg)
        train_dataset.append(ds)
    if len(train_dataset) > 1:
        train_dataset = ConcatDataset(train_dataset)
    else:
        train_dataset = train_dataset[0]
    dev_dataset = []
    for ds_cfg in dataset_cfg['dev_datasets']:
        ds_cfg['params']['semantic_paths'] = args.dev_semantic_file_groups[
            args.global_rank]
        ds_cfg['params']['acoustic_paths'] = args.dev_acoustic_file_groups[
            args.global_rank]
        ds_cfg['params']['max_token_one_batch'] = dataset_cfg[
            'max_token_one_batch']
        ds = instantiate_from_config(ds_cfg)
        dev_dataset.append(ds)
    if len(dev_dataset) > 1:
        dev_dataset = ConcatDataset(dev_dataset)
    else:
        dev_dataset = dev_dataset[0]
    # 这里需要改成新的 sampler
    if args is not None and args.distributed:
        # print("train_dataset.__len__():", train_dataset.__len__())
        train_sampler = DDPSyncSampler(
            size=train_dataset.__len__(),
            seed=seed,
            rank=rank,
            args=args,
            shuffle=True)
        dev_dataset_index_list = list(range(dev_dataset.__len__()))
        dev_sampler = SequentialSampler(dev_dataset_index_list)
        # 注意这里是用 train_sampler 求，表示的是每个卡上的
        # train_iters
        '''
        以下是之前使用 DistributedSampler 观察到的现象:
        self.dataloader['train_iterations']: iter per epoch
        1 卡 -> n 卡，self.max_epochs 不变，self.dataloader['train_iterations'] 为原来的 1/n，单卡显存占用不变
        "self.dataloader['train_iterations'] 为原来的 1/n" 是因为, 1 卡 -> n 卡， len(train_sampler) 变为原来的 1/n
        使用 DDPSyncSampler 后 len(train_sampler) == train_dataset.__len__()
        使用 SequentialSampler 后 len(dev_sampler) == dev_dataset.__len__()
        '''
        # print(f"args.global_rank: {args.global_rank}, len(train_sampler): {len(train_sampler)}, len(dev_sampler): {len(dev_sampler)}")
        train_iters = len(train_sampler) // batch_size
        dev_iters = len(dev_sampler) // batch_size
    else:
        train_sampler = None
        dev_sampler = None
        # 每个 epoch 进行一次
        train_iters = len(train_dataset) // batch_size
        dev_iters = len(dev_dataset) // batch_size
    num_workers = dataset_cfg['num_workers']
    prefetch_factor = dataset_cfg.get('prefetch_factor', 2)
    persistent_workers = True if num_workers > 0 else False
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=train_dataset.collater,
        persistent_workers=persistent_workers,
        # 解决 num_workers > 0 时的 bad value(s) in fds_to_keep 报错
        multiprocessing_context='fork',
        prefetch_factor=prefetch_factor)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        #(dev_sampler is None),
        shuffle=False,
        num_workers=num_workers,
        sampler=dev_sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=train_dataset.collater,
        persistent_workers=persistent_workers,
        multiprocessing_context='fork',
        prefetch_factor=prefetch_factor)
    # 和 dataset.__len__() 一样
    print(
        f"global_rank: {args.global_rank}, train_iters: {train_iters}, dev_iters: {dev_iters}"
    )
    dataload_info = {
        'train_loader': train_loader,
        'dev_loader': dev_loader,
        'train_iterations': train_iters,
        'dev_iterations': dev_iters
    }

    if return_dataset:
        dataload_info['train_dataset'] = train_dataset
        dataload_info['dev_dataset'] = dev_dataset

    return dataload_info
