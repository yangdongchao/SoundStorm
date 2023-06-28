# Fast loader
# it can help to fast read data, so that it can improve the training time.
import torch
from soundstorm.s2.utils.misc import instantiate_from_config
from torch.utils.data import ConcatDataset



def build_dataloader(config, args=None, return_dataset=False):
    dataset_cfg = config['dataloader']
    batch_size = 1
    train_dataset = []
    for ds_cfg in dataset_cfg['train_datasets']:
        # ds_cfg['params']['data_root'] = dataset_cfg.get('data_root', '')
        ds_cfg['params']['semantic_path'] = args.train_semantic_path
        ds_cfg['params']['acoustic_path'] = args.train_acoustic_path
        ds_cfg['params']['max_token_one_batch'] = dataset_cfg['max_token_one_batch']
        ds = instantiate_from_config(ds_cfg)
        train_dataset.append(ds)
    if len(train_dataset) > 1:
        train_dataset = ConcatDataset(train_dataset)
    else:
        train_dataset = train_dataset[0]
    dev_dataset = []
    for ds_cfg in dataset_cfg['dev_datasets']:
        ds_cfg['params']['semantic_path'] = args.dev_semantic_path
        ds_cfg['params']['acoustic_path'] = args.dev_acoustic_path
        ds_cfg['params']['max_token_one_batch'] = dataset_cfg['max_token_one_batch']
        ds = instantiate_from_config(ds_cfg)
        dev_dataset.append(ds)
    if len(dev_dataset) > 1:
        dev_dataset = ConcatDataset(dev_dataset)
    else:
        dev_dataset = dev_dataset[0]

    if args is not None and args.distributed:
        # I add "num_replicas=world_size, rank=rank"
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(
            dev_dataset, shuffle=False)
        train_iters = len(train_sampler) // batch_size
        dev_iters = len(dev_sampler) // batch_size
    else:
        train_sampler = None
        dev_sampler = None
        # 每个 epoch 进行一次
        train_iters = len(train_dataset) // batch_size
        dev_iters = len(dev_dataset) // batch_size
    num_workers = dataset_cfg['num_workers']
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
        # 解决 num_workers>0 时的 bad value(s) in fds_to_keep 报错
        multiprocessing_context='fork')

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
        persistent_workers=persistent_workers)

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
