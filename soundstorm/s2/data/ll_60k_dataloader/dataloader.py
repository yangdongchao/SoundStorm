import copy
import logging
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def pad_2D(inputs, PAD):
    # when each sample in inputs is 2D, this function can be used
    #print('inputs ', inputs.shape)
    def pad(x, max_len):
        #print('x ', x.shape, max_len)
        return F.pad(x, (0, max_len - x.shape[-1]), mode="constant", value=PAD)

    max_len = max(np.shape(x)[-1] for x in inputs)  # 
    output = np.stack([pad(x, max_len) for x in inputs])  # 
    return output


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
    assert batch_length_key in expected_keys
    packed_data = verify_packed_data(packed_data, expected_keys)

    # batchfy 
    batches = batchfy(packed_data, batch_scale, batch_length_key)

    # split train and valid: still keep the ascending length order of batches
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

    if minibatch_debug > 0:  # reduce the size for debug
        train_split = train_split[:min(minibatch_debug, len(train_split))]
        valid_split = valid_split[:min(minibatch_debug, len(valid_split))]

    logging.info(f"train split has {len(train_split)} batches")
    logging.info(f"valid split has {len(valid_split)} batches")

    # Build dataset and sampler
    # "packed_data" is just a pointer so we avoid dupilicated data copy
    train_dataset = Dataset(train_split, packed_data)
    valid_dataset = Dataset(valid_split, packed_data)

    # We must assume that the DDP has been launched so far
    assert torch.distributed.is_initialized() and torch.cuda.is_available()
    rank = dist.get_rank()  # global rank
    train_sampler = DDPSyncSampler(
        size=len(train_split), seed=seed, rank=rank, args=args, shuffle=True)
    valid_sampler = SequentialSampler(list(range(len(valid_split))))

    # Build iterator
    train_iterator = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=n_worker,
        #prefetch_factor=min(5, len(train_split)),
        collate_fn=collate_fn, )

    valid_iterator = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        sampler=valid_sampler,
        num_workers=n_worker,
        #prefetch_factor=min(5, len(valid_split)),
        collate_fn=collate_fn, )

    return train_iterator, valid_iterator


def load_all_data(data_files):
    data_files = data_files.strip().split(",")
    data_dict = {}
    data_keys = []

    for data_file in data_files:
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
    return data_dict, data_keys


def verify_packed_data(packed_data, expected_keys):
    init_number_egs = len(packed_data)  # the length
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


def batchfy(packed_data, batch_scale, batch_length_key):
    # Make sure the batch_length_key feature has the __len__ method
    batch_utts = list(packed_data.keys())
    batch_utts.sort(
        key=lambda x: len(packed_data[x][batch_length_key]))  # 按长度进行排序
    batch_lengths = [len(packed_data[k][batch_length_key])
                     for k in batch_utts]  # 统计长度

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

    if len(batch) > 0:  # 最后一个batch
        batches.append(copy.deepcopy(batch))

    # TODO: maybe report statistics
    return batches


class Dataset(torch.utils.data.Dataset):
    """ Dataset. Each example is exactly a batch """

    def __init__(self, data_split, data_dict):
        self.data_split = data_split
        self.data_dict = data_dict

    def __getitem__(self, index):
        uttids = self.data_split[index]
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


class DDPSyncSampler(object):
    def __init__(self, size, seed, rank, args, shuffle=True):
        self.size = size
        self.seed = seed
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        # Ensure all GPUs have the same number of batches by adding redundant
        # sampling indexes. The redundant samples are not shuffled.
        # Cannot access the "local_rank" variable so it is a bit dirty
        local_rank = args.local_rank  # get local rank
        device = torch.device(f"cuda:{local_rank}")
        size = torch.Tensor([size]).to(device)
        gathered_size = [
            torch.Tensor([size]).to(device)
            for _ in range(dist.get_world_size())
        ]  # 获得总卡数
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
            seg = seq[start:min(self.size, start + chunk_size)]  # 分段
            local_random_order = random.sample(list(range(len(seg))),
                                               len(seg))  # 打乱该段
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
        batched_data = batched_data[0]  # onlu includes one batch
        bsz = len(batched_data)
        batch_prompt_semantics = []  # init the bath seq
        batch_target_semantics = []
        batch_prompt_acoustics = []
        batch_target_acoustics = []
        #uttids = []
        for i, (k, d) in enumerate(batched_data):
            all_semantic_data = d['semantic']  # a numpy data, one dimension
            all_acoustic_data = d['acoustic']
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
            # elif all_semantic_data.shape[0] > 3*50 and all_semantic_data.shape[0] < max_len:
            #     prompt_semantic = all_semantic_data[:300] # choose 3s
            #     target_semantic = all_semantic_data[300:] # 前3s以后，全做为target
            #     prompt_acoustic = all_acoustic_data[:num_q, :300]
            #     target_acoustic = all_acoustic_data[:num_q, 300:] # 
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
            #uttids.append(k)
        prompt_semantics = pad_2D(batch_prompt_semantics,
                                  prompt_semantic_end_id)
        target_semantics = pad_2D(batch_target_semantics,
                                  target_semantic_end_id)
        prompt_acoustics = pad_2D(batch_prompt_acoustics,
                                  prompt_acoustic_end_id)
        target_acoustics = pad_2D(batch_target_acoustics,
                                  target_acoustic_end_id)
        # new_samples = {}
        prompt_semantics = torch.from_numpy(prompt_semantics)
        target_semantics = torch.from_numpy(target_semantics)
        prompt_acoustics = torch.from_numpy(prompt_acoustics)
        target_acoustics = torch.from_numpy(target_acoustics)
        return prompt_semantics, target_semantics, prompt_acoustics, target_acoustics

    return collate_fn
