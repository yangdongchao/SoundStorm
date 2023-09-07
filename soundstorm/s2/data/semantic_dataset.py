import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# BaseDataset code from NATSpeech
'''
数据集构建策略:
(1) prompt 不超过 3s, target 不超过 3s
(2) 若总长度小于 6s, 则 1/2 分给 prompt, 1/2 分给 target.
(3) 分成 se_pro, se-tar, ac_pro, ac_targ 4 个部分返回，每个部分分别 padding 到其 max sample
'''


def pad_2D(inputs: List[torch.Tensor], PAD: int, print_len: bool=False):
    # when each sample in inputs is 2D, this function can be used
    def pad(x: torch.Tensor, max_len: int):
        return F.pad(x, (0, max_len - x.shape[-1]), mode="constant", value=PAD)

    max_len = max(np.shape(x)[-1] for x in inputs)
    input_len = len(inputs)
    if print_len:
        min_len = min(np.shape(x)[-1] for x in inputs)
        print("input_len, max_len, min_len, max_len-min_len:", input_len,
              max_len, min_len, max_len - min_len)
    output = np.stack([pad(x, max_len) for x in inputs])
    return output


class SemanticDataset(torch.utils.data.Dataset):
    def __init__(self,
                 num_quant,
                 semantic_path,
                 acoustic_path,
                 codec_name: str='hificodec',
                 max_token_one_batch: int=10000,
                 semantic_token_nums: int=1000,
                 max_prompt_sec: int=3,
                 max_target_sec: int=10):
        super().__init__()

        self.semantic_data = pd.read_csv(semantic_path, delimiter='\t')
        # get dict
        self.acoustic_data = torch.load(acoustic_path)

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

        if not self.inited:
            # 调用初始化函数
            self.init_batch()
            self.inited = True

    def init_batch(self):
        # this function aims to prepare batch
        # 先根据 semantic_data 的 长度进行排序
        # target 最长设为 10s, prompt 3s, 1s 对应 50 个 token,
        max_token_one_batch = self.max_token_one_batch
        sementic_ls = []
        len_ls = []
        semantic_data_len = len(self.semantic_data)
        acoustic_data_len = len(self.acoustic_data.keys())
        print("semantic_data_len:", semantic_data_len)
        print("acoustic_data_len:", acoustic_data_len)
        for i in range(semantic_data_len):
            # 先依次遍历
            # get str
            semantic_str = self.semantic_data['semantic_audio'][i]
            # get token list
            tmp = [int(idx) for idx in semantic_str.split(' ')]
            sementic_ls.append(tmp)
            len_ls.append(len(tmp))
        # 按列表中元素的值进行排序，并返回元素对应索引序列
        sorted_id = sorted(
            range(len(len_ls)), key=lambda k: len_ls[k], reverse=True)
        start_batch_id = 0
        # 最大长度为 13s
        max_len = self.max_sec * self.hz
        tmp_prompt_semantics = []
        tmp_target_semantics = []
        tmp_prompt_acoustics = []
        tmp_target_acoustics = []
        tmp_tot_tokens = 0
        for i in range(len(sorted_id)):
            # get the index
            index = sorted_id[i]
            # get the semantic 
            # (1, T)
            over_semantic = torch.tensor(sementic_ls[index]).unsqueeze(0)
            # 需要处理 item_name 不在 acoustic_data 中的情况
            item_name = self.semantic_data['item_name'][index]
            try:
                acoustic_str = self.acoustic_data[item_name]
            except Exception:
                print(f"{item_name} not in self.acoustic_data !")
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
                    start_batch_id)] = tmp_prompt_semantics
                self.batch_target_semantics[str(
                    start_batch_id)] = tmp_target_semantics
                self.batch_prompt_acoustics[str(
                    start_batch_id)] = tmp_prompt_acoustics
                self.batch_target_acoustics[str(
                    start_batch_id)] = tmp_target_acoustics
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
                start_batch_id += 1
        # add the last batch
        self.batch_prompt_semantics[str(start_batch_id)] = tmp_prompt_semantics
        self.batch_target_semantics[str(start_batch_id)] = tmp_target_semantics
        self.batch_prompt_acoustics[str(start_batch_id)] = tmp_prompt_acoustics
        self.batch_target_acoustics[str(start_batch_id)] = tmp_target_acoustics

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
