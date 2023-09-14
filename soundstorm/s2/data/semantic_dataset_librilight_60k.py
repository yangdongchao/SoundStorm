import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from soundstorm.s2.data.semantic_dataset import pad_2D

# BaseDataset code from NATSpeech
'''
数据集构建策略:
(1) prompt 不超过 3s, target 不超过 3s
(2) 若总长度小于 6s, 则 1/2 分给 prompt, 1/2 分给 target.
(3) 分成 se_pro, se-tar, ac_pro, ac_targ 4 个部分返回，每个部分分别 padding 到其 max sample
'''


class SemanticDataset(torch.utils.data.Dataset):
    def __init__(self,
                 num_quant,
                 semantic_paths,
                 acoustic_paths,
                 codec_name: str='hificodec',
                 max_token_one_batch: int=10000,
                 semantic_token_nums: int=1000,
                 max_prompt_sec: int=3,
                 max_target_sec: int=10):
        super().__init__()

        self.semantic_data_dict = dict()
        self.acoustic_data_dict = dict()

        semantic_files = semantic_paths
        acoustic_files = acoustic_paths
        print("semantic_files:", semantic_files)
        s_st = time.time()
        for semantic_file in semantic_files:
            name_list = semantic_file.split("/")
            # semantic_token_0_3.npy -> 0_3
            rank_name = '_'.join(name_list[-1].split('.')[0].split('_')[-2:])
            # small/medium/large/duplicate_0_3
            key_name = f'{name_list[-3]}_{rank_name}'
            self.semantic_data_dict[key_name] = np.load(
                semantic_file, allow_pickle=True).item()
        print(f"load semantic_files done, cost {round(time.time()-s_st, 2)}s")
        a_st = time.time()
        for acoustic_file in acoustic_files:
            name_list = acoustic_file.split("/")
            rank_name = '_'.join(name_list[-1].split('.')[0].split('_')[-2:])
            key_name = f'{name_list[-4]}_{rank_name}'
            self.acoustic_data_dict[key_name] = torch.load(acoustic_file)
        print(f"load acoustic_files done, cost {round(time.time()-a_st, 2)}s")

        self.num_quant = 4 if codec_name == 'hificodec' else num_quant
        # 16000 / 320 = 50
        self.hz = 50  # 分辨率

        self.max_prompt_sec = max_prompt_sec
        self.max_target_sec = max_target_sec
        self.max_sec = self.max_prompt_sec + self.max_target_sec

        # NOTE by yuantian: same as SemanticTokenizer.dim_codebook
        self.semantic_token_nums = semantic_token_nums
        # self.prompt_semantic_start_id = self.semantic_token_nums
        self.prompt_semantic_end_id = self.semantic_token_nums + 1
        # self.target_semantic_start_id = self.semantic_token_nums + 2
        self.target_semantic_end_id = self.semantic_token_nums + 3
        # NOTE by yuantian: N in codec
        self.acoustic_token_nums = 1024
        self.prompt_acoustic_eos = self.acoustic_token_nums
        self.target_acoustic_eos = self.acoustic_token_nums + 1

        self.batch_prompt_semantics = {}
        self.batch_target_semantics = {}
        self.batch_prompt_acoustics = {}
        self.batch_target_acoustics = {}

        # 一个 batch 最多多少个 token
        self.max_token_one_batch = max_token_one_batch
        self.inited = False
        self.start_batch_id = 0
        self.total_semantic_data_len = 0
        self.total_acoustic_data_len = 0
        # 这里是否有必要写成多并发？8 卡 60k 每个 rank semantic_data_dict 长度是 3 ~ 4
        if not self.inited:
            # 调用初始化函数
            for key_name in self.semantic_data_dict.keys():
                i_st = time.time()
                self.init_batch(key_name)
                print(
                    f"init_batch of {key_name} done, cost {round(time.time()-i_st, 2)}s"
                )
            self.inited = True
            print("self.total_semantic_data_len:", self.total_semantic_data_len)
            print("self.total_acoustic_data_len:", self.total_acoustic_data_len)
            # 组成了多少个 batch, data_len: self.__len__() ~= 20: 1
            # 即 max_token_one_batch = 3k 时约 20 句组成一个 batch (平均, 因为句子先按照长度排序了)
            # print("self.__len__():",self.__len__())

    def init_batch(self, key_name: str):
        # this function aims to prepare batch
        # 先根据 semantic_data 的 长度进行排序
        # target 最长设为 10s, prompt 3s, 1s 对应 50 个 token,
        if key_name not in self.acoustic_data_dict.keys():
            print(f'{key_name} not in self.acoustic_data_dict')
            return None

        semantic_data = self.semantic_data_dict[key_name]
        acoustic_data = self.acoustic_data_dict[key_name]
        max_token_one_batch = self.max_token_one_batch
        sementic_ls = []
        len_ls = []
        semantic_data_len = len(semantic_data)
        acoustic_data_len = len(acoustic_data.keys())

        self.total_semantic_data_len += semantic_data_len
        self.total_acoustic_data_len += acoustic_data_len

        semantic_data = OrderedDict(
            sorted(
                semantic_data.items(), key=lambda x: len(x[1]), reverse=True))

        # 最大长度为 13s
        max_len = self.max_sec * self.hz
        tmp_prompt_semantics = []
        tmp_target_semantics = []
        tmp_prompt_acoustics = []
        tmp_target_acoustics = []
        tmp_tot_tokens = 0
        for item_name, semantic_ids in semantic_data.items():
            # get the semantic 
            # (1, T)
            over_semantic = torch.tensor(semantic_ids).unsqueeze(0)
            # 需要处理 item_name 不在 acoustic_data 中的情况
            try:
                acoustic_str = acoustic_data[item_name]
            except Exception:
                # too many print for LibriTTS large
                # print(f"{item_name} not in acoustic_data !")
                continue
            # only keep the first num_quant codebooks
            # 这里表明 acoustic_token 的存储方式是 (C, T)
            over_acoustic = acoustic_str[:self.num_quant, ...]
            over_semantic_len = over_semantic.shape[1]
            if over_semantic_len >= max_len:
                # 若音频长度大于 13s，则考虑切成 3 + 10, prompt 3s, target 10s
                prompt_len = self.max_prompt_sec * self.hz
                targen_len = self.max_target_sec * self.hz
                # 先随机选一个 prompt 起始点，max 为最后 13s
                # 总长度剪去 13s
                max_prompt_index = over_semantic_len - max_len
                left_start = random.randint(0, max_prompt_index)
                prompt_end = left_start + prompt_len
                prompt_semantic = over_semantic[:, left_start:prompt_end]
                prompt_acoustic = over_acoustic[:, left_start:prompt_end]
                # 往后数 10s
                target_semantic = over_semantic[:, prompt_end:prompt_end +
                                                targen_len]
                target_acoustic = over_acoustic[:, prompt_end:prompt_end +
                                                targen_len]
            # 如果长度大于 6s 小于 13s
            elif over_semantic_len > 2 * self.max_prompt_sec * self.hz and over_semantic_len < max_len:
                # 3s 的 prompt, 其后全为 target
                prompt_len = self.max_prompt_sec * self.hz
                prompt_semantic = over_semantic[:, :prompt_len]
                prompt_acoustic = over_acoustic[:, :prompt_len]
                # 前 3s 以后，全做为 target  
                target_semantic = over_semantic[:, prompt_len:]
                target_acoustic = over_acoustic[:, prompt_len:]
            else:
                # 小于 6s，直接平均分
                mid_id = int(over_semantic_len / 2)
                # choose 3s
                prompt_semantic = over_semantic[:, :mid_id]
                prompt_acoustic = over_acoustic[:, :mid_id]
                # 前 3s 以后，全做为 target
                target_semantic = over_semantic[:, mid_id:]
                target_acoustic = over_acoustic[:, mid_id:]
            # 计算当前数据的 token 数量
            cal_num = prompt_semantic.shape[1] + target_semantic.shape[
                1] + prompt_acoustic.shape[1] + target_acoustic.shape[1]
            if tmp_tot_tokens + cal_num < max_token_one_batch:
                # 若还没满一个 batch ,继续添加
                # shape: (1, 150)
                tmp_prompt_semantics.append(prompt_semantic)
                tmp_target_semantics.append(target_semantic)
                tmp_prompt_acoustics.append(prompt_acoustic)
                tmp_target_acoustics.append(target_acoustic)
                # 添加当前 batch 的 token 数量
                tmp_tot_tokens += cal_num
            else:
                # 若已满一个 batch
                # save batch
                self.batch_prompt_semantics[str(
                    self.start_batch_id)] = tmp_prompt_semantics
                self.batch_target_semantics[str(
                    self.start_batch_id)] = tmp_target_semantics
                self.batch_prompt_acoustics[str(
                    self.start_batch_id)] = tmp_prompt_acoustics
                self.batch_target_acoustics[str(
                    self.start_batch_id)] = tmp_target_acoustics
                # clear previous step
                tmp_prompt_semantics = []
                tmp_target_semantics = []
                tmp_prompt_acoustics = []
                tmp_target_acoustics = []
                # 重置为 0
                tmp_tot_tokens = 0
                # add new batch
                tmp_prompt_semantics.append(prompt_semantic)
                tmp_target_semantics.append(target_semantic)
                tmp_prompt_acoustics.append(prompt_acoustic)
                tmp_target_acoustics.append(target_acoustic)
                tmp_tot_tokens += cal_num
                self.start_batch_id += 1
        # add the last batch
        self.batch_prompt_semantics[str(
            self.start_batch_id)] = tmp_prompt_semantics
        self.batch_target_semantics[str(
            self.start_batch_id)] = tmp_target_semantics
        self.batch_prompt_acoustics[str(
            self.start_batch_id)] = tmp_prompt_acoustics
        self.batch_target_acoustics[str(
            self.start_batch_id)] = tmp_target_acoustics

    def __len__(self):
        return len(self.batch_prompt_semantics)

    def __getitem__(self, index):
        prompt_semantic = self.batch_prompt_semantics[str(index)]
        target_semantic = self.batch_target_semantics[str(index)]
        prompt_acoustic = self.batch_prompt_acoustics[str(index)]
        target_acoustic = self.batch_target_acoustics[str(index)]
        sample = {}
        sample['prompt_semantic'] = prompt_semantic
        sample['target_semantic'] = target_semantic
        sample['prompt_acoustic'] = prompt_acoustic
        sample['target_acoustic'] = target_acoustic
        return sample

    def collater(self, samples):
        # 为什么只取 第 0 个? => 因为 samples 是 list 长度一直是 1, batch_size must be 1 here
        # prompt_semantics 里面是 n 个 tensor, n 的大小不固定
        # len(prompt_semantics) = 100 ，表示 batch_size = 100, batch_size 是不固定的
        sample = samples[0]
        prompt_semantics = sample['prompt_semantic']
        target_semantics = sample['target_semantic']
        prompt_acoustics = sample['prompt_acoustic']
        target_acoustics = sample['target_acoustic']
        # in this version, we do not use pading token any more, instead, we use eos token
        # 一个 batch 里面按照最长的补 0
        prompt_semantics = pad_2D(prompt_semantics, self.prompt_semantic_end_id)
        target_semantics = pad_2D(target_semantics, self.target_semantic_end_id)
        prompt_acoustics = pad_2D(prompt_acoustics, self.prompt_acoustic_eos)
        # 用 1025 补零
        target_acoustics = pad_2D(target_acoustics, self.target_acoustic_eos)
        # mask 住 target_acoustics 的补 0 部分
        x_mask = (target_acoustics == self.target_acoustic_eos)
        new_samples = {}
        # (B, 1, T), B, T 动态
        new_samples['prompt_semantics'] = torch.from_numpy(prompt_semantics)
        new_samples['target_semantics'] = torch.from_numpy(target_semantics)
        new_samples['prompt_acoustics'] = torch.from_numpy(prompt_acoustics)
        # (B, 4, T), B, T 动态
        new_samples['target_acoustics'] = torch.from_numpy(target_acoustics)
        new_samples['x_mask'] = torch.from_numpy(x_mask[:, 0, :])
        return new_samples


class SequentialSampler(object):
    def __init__(self, sequence):
        self.seq = sequence

    def __iter__(self):
        # 直接返回顺序的 batch index, 不打乱
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
        # DataSet 中总 batch 数，
        self.size = size
        self.seed = seed
        # global_rank, 对完整数据集也是按照 global_rank 数划分的
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        # ❗️ Ensure all GPUs have the same number of batches by adding redundant
        # sampling indexes. The redundant samples are not shuffled.
        # Cannot access the "local_rank" variable so it is a bit dirty
        # get local rank
        # 此处 local_rank 主要是为了设置 device
        local_rank = args.local_rank
        device = torch.device(f"cuda:{local_rank}")
        size = torch.Tensor([size]).to(device)
        # 获得总卡数
        # dist.get_world_size 对应 dist.get_rank() 是 global_rank
        # 为 global_rank 的每个 rank 都造一个 size
        gathered_size = [size for _ in range(dist.get_world_size())]
        # list 里面每个值依旧一样
        torch.distributed.all_gather(gathered_size, size)
        # 计算当前 gpu 需要 padding 的 batch 数量
        # always 0
        self.pad_number = int(max(gathered_size).item() - size.item())
        # 对 batch index 进行一系列操作, 但是长度还是与 dataset.__len__() 保持一致
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
            local_random_order = random.sample(list(range(len(seg))), len(seg))
            seg = [seg[i] for i in local_random_order]
            seq[start:min(self.size, start + chunk_size)] = seg
            start += len(seg)

        # even after this shuffle, the batch lengths across GPUs 
        # are very similar
        # 上面是段内打乱，这里是段和段的顺序打乱
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(seq)

        if self.pad_number > 0:
            seq = list(range(self.pad_number)) + seq
        # list(int), 每个值表示 batch index
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    semantic_dirs_1 = '/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_librilight/small/train/'
    acoustic_dirs_1 = '/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_librilight/small/train/acoustic_token/'
    start_build_time = time.time()
    dataset = SemanticDataset(
        semantic_paths=[
            semantic_dirs_1 + 'semantic_token_0_3.npy',
            semantic_dirs_1 + 'semantic_token_1_3.npy',
            semantic_dirs_1 + 'semantic_token_2_3.npy'
        ],
        acoustic_paths=[
            acoustic_dirs_1 + 'hificodec_0_3.pth',
            acoustic_dirs_1 + 'hificodec_1_3.pth',
            acoustic_dirs_1 + 'hificodec_2_3.pth'
        ],
        num_quant=4, )
    batch_size = 12
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collater,
        shuffle=False)

    print(f"time of build dataloader: {time.time() - start_build_time}")
    for i, batch in enumerate(dataloader):
        if i == 0:
            print('batch["prompt_semantics"]:', batch["prompt_semantics"].shape)
            print('batch["target_semantics"]:', batch["target_semantics"].shape)
            print('batch["prompt_acoustics"]:', batch["prompt_acoustics"].shape)
            print('batch["target_acoustics"]:', batch["target_acoustics"].shape)
