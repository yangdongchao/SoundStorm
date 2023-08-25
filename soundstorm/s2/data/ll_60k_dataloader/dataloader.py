import copy
import logging
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


from soundstorm.s2.data.semantic_dataset import pad_2D


# 这个函数的作用类似于 build.py
def build_train_and_valid_data_iterator(
        args,
        packed_data,
        collate_fn,
        batch_scale=1000,
        batch_length_key='semantic',
        valid_batches=10,
        n_worker=0,
        seed=999,
        minibatch_debug=-1, ):

    # Make sure this is identical to each GPU
    random.seed(seed)

    # load and verify
    packed_data, expected_keys = load_all_data(packed_data)
    # assert batch_length_key in expected_keys
    packed_data = verify_packed_data(packed_data, expected_keys)

    # batchfy 
    # 按照 semantic 的长度进行排序
    # [[utt1, utt2], [utt3, utt4], [utt5, utt6]]
    batches = batchfy(packed_data, batch_scale, batch_length_key)

    # split train and valid: still keep the ascending length order of batches
    # 分 train 和 valid
    assert valid_batches * 10 < len(
        batches), f"valid_batches should be smaller than {len(batches) / 10}"
    valid_index = {
        k: None
        for k in random.sample(range(len(batches)), valid_batches)
    }  # 随机采样valid_batches个 
    valid_split = [batches[i] for i in range(len(batches)) if i in valid_index]
    train_split = [
        batches[i] for i in range(len(batches)) if i not in valid_index
    ]
    # reduce the size for debug
    if minibatch_debug > 0: 
        train_split = train_split[:min(minibatch_debug, len(train_split))]
        valid_split = valid_split[:min(minibatch_debug, len(valid_split))]

    logging.info(f"train split has {len(train_split)} batches")
    logging.info(f"valid split has {len(valid_split)} batches")

    # Build dataset and sampler
    # "packed_data" is just a pointer so we avoid dupilicated data copy
    # train_split: [[utt1, utt2], [utt3, utt4], [utt5, utt6]]

    # 获取 dataset
    train_dataset = Dataset(train_split, packed_data)
    valid_dataset = Dataset(valid_split, packed_data)

    # We must assume that the DDP has been launched so far
    assert torch.distributed.is_initialized() and torch.cuda.is_available()
    # global rank
    rank = dist.get_rank()
    # 重点看下这里
    # sampler 传入的 len(train_split) 而不是 train_dataset
    # DistributedSampler 传入的是 dataset
    #  len(train_split) == train_dataset.__len__
    train_sampler = DDPSyncSampler(
        size=len(train_split), seed=seed, rank=rank, args=args, shuffle=True)
    # 为啥 valid_sampler 不和 train_sampler 用一样的
    valid_sampler = SequentialSampler(list(range(len(valid_split))))

    # 获取 dataloader

    # Build iterator
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=n_worker,
        #prefetch_factor=min(5, len(train_split)),
        collate_fn=collate_fn, )

    dev_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        sampler=valid_sampler,
        num_workers=n_worker,
        collate_fn=collate_fn, )

    train_iters = len(train_dataset) // batch_size
    dev_iters = len(dev_dataset) // batch_size
    
    dataload_info = {
        'train_loader': train_loader,
        'dev_loader': dev_loader,
        'train_iterations': train_iters,
        'dev_iterations': dev_iters
    }

    return dataload_info

# 这个 rank 对应的数据路径
'''
ontent = 'semantic:' + '/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/dongchao/code5/SpearTTS_32gpu_test/data_infor/semantic/gpu_' + str(
        args.global_rank) + '.pth'
content = content + ",acoustic:" + '/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/dongchao/code5/SpearTTS_32gpu_test/data_infor/acoustic/gpu_' + str(
    args.global_rank) + '.pth'
'''
def load_all_data(data_files):
    # 一个字符串，用逗号分割 
    # => 我们可以传2 个 list semantic_list, acoustic_list, 然后对应的 split 存到对应的字典
    data_files = data_files.strip().split(",")
    data_dict = {}
    data_keys = []

    for data_file in data_files:
        # data_key 表明是 semantic 还是 acoustic
        data_key, data_file = data_file.strip().split(':')
        data_keys.append(data_key)
        logging.info(f'loading data key {data_key} from file {data_file}')
        # name, data_file, such as phone or acoustic
        if data_file.endswith(".pth"):
            # .pt file in dict format: {uttid: data}
            this_data = torch.load(data_file)
        else:
            # text file in line format: <uttid> <data sequence>
            this_data = open(data_file, encoding='utf-8').readlines()
            this_data = [l.strip().split() for l in this_data]
            this_data = {l[0]: l[1:] for l in this_data}
        print('this_data.items() ', len(this_data.items()))
        for k, v in this_data.items():
            if k not in data_dict:
                data_dict[k] = {}

            # to be compatible with previous packed_data. 
            # very ugly. should be removed in the future
            if isinstance(v, dict):
                try:
                    v = v[data_key]
                except Exception:
                    pass
            data_dict[k][data_key] = v
    '''
    data_dict:
    { 'utt_id':
        semantic: .....
        acoustic: .....
    }
    data_keys:
    ['semantic', 'acoustic']
    '''
    return data_dict, data_keys

# 确保每个 utt_id 都有 expected_keys ['semantic', 'acoustic'] 的内容, 有问题的会删除
def verify_packed_data(packed_data, expected_keys):
    # the length
    init_number_egs = len(packed_data)  
    uttids = list(packed_data.keys())
    for uttid in uttids:
        for key in expected_keys:
            if key not in packed_data[uttid]:
                # 若key不全
                del packed_data[uttid]
                break

            if packed_data[uttid][key] is None:
                # 若特征为none
                del packed_data[uttid]
                break

    final_number_egs = len(packed_data)

    if final_number_egs < init_number_egs:
        # 检测有多少有用的数据
        logging.warning(
            f'Number of utterances has been reduced from {init_number_egs} to {final_number_egs} since some examples are invalid.'
        )

    if final_number_egs == 0:
        logging.error('No utterances are valid after data verification')

    logging.info(
        f'Finish data verification. We have {final_number_egs} utterances in total.'
    )
    return packed_data

# 因为是在 collect_fn 才切片的，和原来的不一样
# 所以是在这里统计时长
# l = min(l, 500) 可以换成 20s
# ❗️ 能否直接用 semantic_dataset 里面的 init_batch 的写法
def batchfy(packed_data, batch_scale, batch_length_key):
    # batch_scale 一个 batch 里面 semantic token 的长度
    # Make sure the batch_length_key feature has the __len__ method
    # 所有的 utt_id 列成一个列表
    batch_utts = list(packed_data.keys())
    # 原来是按照 max_token_one_batch 塞进 batch
    # 按长度 semantic 长度排序
    batch_utts.sort(
        key=lambda x: len(packed_data[x][batch_length_key])) 
    # 统计长度 
    batch_lengths = [len(packed_data[k][batch_length_key])
                     for k in batch_utts]

    # Only take care of the uttid rather than the whole example
    batches, batch, summed_tokens = [], [], 0
    for utt, l in zip(batch_utts, batch_lengths):
        l = min(l, 500)  # 不管音频多长，最后使用时，最多只使用13s,因此只需要考虑 10*50 
        if l + summed_tokens > batch_scale:
            assert len(
                batch) > 0, f"batch_tokens should be larger: {batch_scale}"
            batches.append(copy.deepcopy(batch))  # 保存当前batch
            batch, summed_tokens = [], 0  #重置
        summed_tokens += l
        batch.append(utt)
    # 最后一个batch
    if len(batch) > 0:
        batches.append(copy.deepcopy(batch))

    # TODO: maybe report statistics
    #  list [batch, batch, batch, batch]
    return batches


class Dataset(torch.utils.data.Dataset):
    """ Dataset. Each example is exactly a batch """

    def __init__(self, data_split, data_dict):
        self.data_split = data_split
        self.data_dict = data_dict

    def __getitem__(self, index):
        # train_split: [[utt1, utt2], [utt3, utt4], [utt5, utt6]]
        # uttids: [utt1, utt2]
        # 原始的是取 self.batch_prompt_semantics 类的 index
        # 原始的凑 batch 是在 
        # 现在是在 build_train_and_valid_data_iterator 里面调用别的函数
        uttids = self.data_split[index]
        '''
        [
            （utt_id, item)
             item 是个字典，有如下 2 个 key
                semantic: .....
                acoustic: ..... 
        ]
        '''
        return [(uttid, self.data_dict[uttid]) for uttid in uttids]

    def __len__(self):
        return len(self.data_split)


class SequentialSampler(object):
    def __init__(self, sequence):
        self.seq = sequence

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def refresh(self):
        pass

# 如果直接用 torch.utils.data.distributed.DistributedSampler, 他会认为数据是给所有 rank 的
# 那么如果有 n 块卡，这块卡只会取 1/n 的数据，而其他卡看不到剩下 (n-1)/n 的数据，就会浪费
# 所以需要特别改造 DDPSyncSampler, ❗️保证这个 rank 能看到所有的数据
class DDPSyncSampler(object):
    def __init__(self, size, seed, rank, args, shuffle=True):
        self.size = size
        self.seed = seed
        # global_rank
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        # Ensure all GPUs have the same number of batches by adding redundant
        # sampling indexes. The redundant samples are not shuffled.
        # Cannot access the "local_rank" variable so it is a bit dirty
        # get local rank
        local_rank = args.local_rank  
        device = torch.device(f"cuda:{local_rank}")
        size = torch.Tensor([size]).to(device)
        # 获得总卡数
        # dist.get_world_size 对应 dist.get_rank() 是 global_rank
        gathered_size = [
            torch.Tensor([size]).to(device)
            for _ in range(dist.get_world_size())
        ]  
        torch.distributed.all_gather(gathered_size, size)
        self.pad_number = int(
            max(gathered_size).item() - size.item())  # 计算当前gpu需要padding的batch数量
        self.refresh()

    def refresh(self):
        seq = list(range(self.size))
        # introduce local randomness by local random shuffling
        # otherwise each global batch will be identical across epochs
        chunk_size, start = 10, 0
        random.seed(self.rank + self.seed + self.epoch)
        while start < self.size:
            # 分段
            seg = seq[start:min(self.size, start + chunk_size)]
            # 打乱该段
            local_random_order = random.sample(list(range(len(seg))),
                                               len(seg))
            seg = [seg[i] for i in local_random_order]
            seq[start:min(self.size, start + chunk_size)] = seg
            start += len(seg)

        # even after this shuffle, the batch lengths across GPUs 
        # are very similar
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(seq)

        if self.pad_number > 0:
            seq = list(range(self.pad_number)) + seq

        self.seq = seq
        self.epoch += 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def get_state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'seed': self.seed,
        }
        return state_dict

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

# 这里的 Pad2D 在 collect_fn 做
# 原来的 collect_fn 是 dataset 的一个子函数，可以用 DataSet 的全局对象
def get_collect_cn_function(max_len=(7 * 50), choice=None):
    # we can define multiple data process method in this function
    semantic_token_nums = 1000
    prompt_semantic_end_id = semantic_token_nums + 1
    target_semantic_end_id = semantic_token_nums + 3
    acoustic_token_nums = 1024  # 
    #self.acoustic_pad_id = self.acoustic_token_nums + 1 
    prompt_acoustic_end_id = acoustic_token_nums + 1
    target_acoustic_end_id = acoustic_token_nums + 3
    num_q = 3

    def collate_fn(batched_data):
        # onlu includes one batch
        '''
        batch_data = [ 
            [
            （utt_id, item)
             item 是个字典，有如下 2 个 key
                semantic: .....
                acoustic: ..... 
            ] 
        ]
        '''
        # 最外层的列表表示 batch_size
        
        batched_data = batched_data[0] 
        bsz = len(batched_data)
        # init the bath seq
        batch_prompt_semantics = []  
        batch_target_semantics = []
        batch_prompt_acoustics = []
        batch_target_acoustics = []
        #uttids = []
        # k: utt_id
        # d: item is a dict
        # 这里是取每个 utt_id 然后 append 到对应的列表里 => 
        # ❓这件事和在 init_batch 函数里面做有什么区别？
        # 难道按照 rank load 数据 split 就不能有个全局的 self.batch_prompt_semantics？

        for i, (k, d) in enumerate(batched_data):
            # a numpy data, one dimension
            all_semantic_data = d['semantic']  
            all_acoustic_data = d['acoustic']
            # 这里的切片逻辑换成 semantic_dataset 里面的
            if all_semantic_data.shape[0] > max_len:  # 若长度大于7s
                max_prompt_index = all_semantic_data.shape[
                    0] - max_len  # 总长度剪去13s
                left_start = random.randint(0, max_prompt_index)  # 
                prompt_semantic = all_semantic_data[left_start:left_start +
                                                    150]  # choose 3s
                target_semantic = all_semantic_data[left_start + 150:left_start
                                                    + 150 + 200]  # 往后数4s
                prompt_acoustic = all_acoustic_data[:num_q, left_start:
                                                    left_start + 150]
                target_acoustic = all_acoustic_data[:num_q, left_start + 150:
                                                    left_start + 150 + 200]  # 
         
            else:
                mid_id = int(all_semantic_data.shape[0] / 2)
                prompt_semantic = all_semantic_data[:mid_id]  # choose 3s
                target_semantic = all_semantic_data[mid_id:]  # 前3s以后，全做为target
                prompt_acoustic = all_acoustic_data[:num_q, :mid_id]
                target_acoustic = all_acoustic_data[:num_q, mid_id:]  # 

            prompt_semantic = torch.from_numpy(prompt_semantic).unsqueeze(0)
            target_semantic = torch.from_numpy(target_semantic).unsqueeze(0)
            prompt_acoustic = torch.from_numpy(prompt_acoustic)
            target_acoustic = torch.from_numpy(target_acoustic)


            batch_prompt_semantics.append(prompt_semantic)
            batch_target_semantics.append(target_semantic)
            batch_prompt_acoustics.append(prompt_acoustic)
            batch_target_acoustics.append(target_acoustic)


        # 之前的 collect 从这里开始
        # batch_target_acoustics 就是 samples[0]
        prompt_semantics = pad_2D(batch_prompt_semantics,
                                  prompt_semantic_end_id)
        target_semantics = pad_2D(batch_target_semantics,
                                  target_semantic_end_id)
        prompt_acoustics = pad_2D(batch_prompt_acoustics,
                                  prompt_acoustic_end_id)
        target_acoustics = pad_2D(batch_target_acoustics,
                                  target_acoustic_end_id)
        
        # mask 住 target_acoustics 的补 0 部分
        x_mask = (target_acoustics == self.target_acoustic_eos)

        # 这里需要换成一个 dict
        new_samples = {}
        new_samples['prompt_semantics'] = torch.from_numpy(prompt_semantics)
        new_samples['target_semantics'] = torch.from_numpy(target_semantics)
        new_samples['prompt_acoustics'] = torch.from_numpy(prompt_acoustics)
        # (B, 4, T), B, T 动态
        new_samples['target_acoustics'] = torch.from_numpy(target_acoustics)
        new_samples['x_mask'] = torch.from_numpy(x_mask[:, 0, :])

        return new_samples

    return collate_fn
