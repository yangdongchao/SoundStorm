import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# BaseDataset code from NATSpeech
'''
数据集构建策略:
(1) prompt 不超过3s, target不超过3s
(2) 若总长度小于6s, 则 1/2分给prompt, 1/2分给 target.
(3) 分成 se_pro, se-tar, ac_pro, ac_targ 4个部分返回，每个部分分别padding到其max sample
'''


def pad_2D(inputs, PAD):
    # when each sample in inputs is 2D, this function can be used
    # print('inputs ', inputs.shape)
    def pad(x, max_len):
        #print('x ', x.shape, max_len)
        return F.pad(x, (0, max_len - x.shape[-1]), mode="constant", value=PAD)

    max_len = max(np.shape(x)[-1] for x in inputs)  # 
    output = np.stack([pad(x, max_len) for x in inputs])  # 
    return output


def pad_1D(inputs, PAD):
    def pad(x, max_len):
        return F.pad(x, (0, max_len - x.shape[-1]), mode="constant", value=PAD)

    max_len = max(np.shape(x)[0] for x in inputs)
    output = np.stack([pad(x, max_len) for x in inputs])  # 
    return output


class SemanticDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            num_quant,
            max_length=(250, 250), 
            semantic_path,
            acoustic_path,
            ):
        super().__init__()

        self.semantic_data = pd.read_csv(semantic_path, delimiter='\t')
        # get dict
        self.acoustic_data = torch.load(acoustic_path)

        self.max_length = max_length
        self.num_quant = num_quant
        # 16000 / 320 = 50
        self.hz = 50  # 分辨率
        self.segment_size = 3  # 默认使用3s一个segments
        self.sizes = [
            len(self.semantic_data['tgt_audio'][i].split(' '))
            for i in range(len(self.semantic_data))
        ]
        self.semantic_token_nums = 1000
        self.prompt_semantic_start_id = self.semantic_token_nums
        self.prompt_semantic_end_id = self.semantic_token_nums + 1
        self.target_semantic_start_id = self.semantic_token_nums + 2
        self.target_semantic_end_id = self.semantic_token_nums + 3
        self.acoustic_token_nums = 1024  # 
        self.prompt_acoustic_eos = self.acoustic_token_nums
        self.target_acoustic_eos = self.acoustic_token_nums + 1
        # self.prompt_acoustic_start_id = self.acoustic_token_nums 
        # self.prompt_acoustic_end_id = self.acoustic_token_nums + 1
        # self.target_acoustic_start_id = self.acoustic_token_nums + 2
        # self.target_acoustic_end_id = self.acoustic_token_nums + 3
        # 一个 batch 最多 10000 个 token
        self.max_token_one_batch = 10000
        # 调用初始化函数
        self.init_my_batch()
        print('data size:', self.__len__())

    def init_my_batch(self):
        # this function aims to prepare batch
        # 一个 batch 的总 token 数量设为 5600 
        # 先根据 semantic_data 的 长度进行排序
        # target 最长设为 10s, prompt 3s, 1s 对应 50 个 token,
        # 因此，若使用 10s，则一条数据有 500*3 + 500 + 150+ 150 = 2300 个 token, 则只能放 2 条数据
        max_token_one_batch = self.max_token_one_batch
        sementic_ls = []
        len_ls = []
        for i in range(len(self.semantic_data)):
            # 先依次遍历
            # get str
            semantic_str = self.semantic_data['semantic_audio'][i]
            # get token list
            tmp = [int(idx) for idx in semantic_str.split(' ')]
            sementic_ls.append(tmp)
            len_ls.append(len(tmp))
        # 按列表 a 中元素的值进行排序，并返回元素对应索引序列
        sorted_id = sorted(
            range(len(len_ls)), key=lambda k: len_ls[k], reverse=True)
        start_batch_id = 0
        self.batch_prompt_semantics = {}
        self.batch_target_semantics = {}
        self.batch_prompt_acoustics = {}
        self.batch_target_acoustics = {}
        max_len = 13 * 50
        tmp_prompt_semantics = []
        tmp_target_semantics = []
        tmp_prompt_acoustics = []
        tmp_target_acoustics = []
        tmp_tot_tokens = 0
        for i in range(len(sorted_id)):
            # get the index
            index = sorted_id[i]
            # get the semantic 
            over_semantic = torch.tensor(sementic_ls[index]).unsqueeze(0)
            item_name = self.semantic_data['item_name'][index]
            acoustic_str = self.acoustic_data[item_name]
            # only keep the first 3 codebooks
            over_acoustic = torch.tensor(
                acoustic_str[:self.num_quant, ...]).squeeze(1)
            if over_semantic.shape[1] > max_len:
                # 若音频长度大于13s，则考虑切成3+10
                # 先随机选一个prompt对起始点，max为最后13s
                # 总长度剪去13s
                max_prompt_index = over_semantic.shape[1] - max_len
                left_start = random.randint(0, max_prompt_index)
                # choose 3s 
                prompt_semantic = over_semantic[:, left_start:left_start + 150]
                # 往后数 10s
                target_semantic = over_semantic[:, left_start + 150:left_start +
                                                150 + 500]
                prompt_acoustic = over_acoustic[:, left_start:left_start + 150]
                target_acoustic = over_acoustic[:, left_start + 150:left_start +
                                                150 + 500]

            elif over_semantic.shape[1] > 6 * 50 and over_semantic.shape[
                    1] < max_len:
                # 能保证 3s 的 prompt -> NOTE by yuantian: ❎ 可是这里 300 是 6s 啊？
                # choose 3s
                prompt_semantic = over_semantic[:, :300]
                # 前3s以后，全做为target  
                target_semantic = over_semantic[:, 300:]
                prompt_acoustic = over_acoustic[:, :300]
                target_acoustic = over_acoustic[:, 300:]
                cal_num = prompt_semantic.shape[1] + target_semantic.shape[
                    1] + prompt_acoustic.shape[1] + target_acoustic.shape[1]
            else:
                # 小于 6s，直接平均分
                mid_id = int(over_semantic.shape[1] / 2)
                # choose 3s
                prompt_semantic = over_semantic[:, :mid_id]
                # 前 3s 以后，全做为 target
                target_semantic = over_semantic[:, mid_id:]
                prompt_acoustic = over_acoustic[:, :mid_id]
                target_acoustic = over_acoustic[:, mid_id:]
            # 计算当前数据的 token 数量
            cal_num = prompt_semantic.shape[1] + target_semantic.shape[
                1] + prompt_acoustic.shape[1] + target_acoustic.shape[1]
            if tmp_tot_tokens + cal_num < max_token_one_batch:
                # 若还没满一个batch,继续添加
                tmp_prompt_semantics.append(prompt_semantic)
                tmp_target_semantics.append(target_semantic)
                tmp_prompt_acoustics.append(prompt_acoustic)
                tmp_target_acoustics.append(target_acoustic)
                # 添加当前batch的token数量
                tmp_tot_tokens = tmp_tot_tokens + cal_num
            else:
                # 若已满一个batch
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
        # print('batch_prompt_semantics ', len(self.batch_prompt_semantics))
        # print(len(self.batch_prompt_semantics['0']))
        # print(len(self.batch_prompt_semantics['10000']))
        # assert 1==2
        # for i in range(len(self.batch_prompt_semantics)):
        #     print(len(self.batch_prompt_semantics[str(i)]))
        # assert 1==2

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
        # print(len(samples))
        prompt_semantics = samples[0]['prompt_semantic']
        target_semantics = samples[0]['target_semantic']
        prompt_acoustics = samples[0]['prompt_acoustic']
        target_acoustics = samples[0]['target_acoustic']
        # in this version, we do not use pading token any more, instead, we use eos token
        prompt_semantics = pad_2D(prompt_semantics, self.prompt_semantic_end_id)
        target_semantics = pad_2D(target_semantics, self.target_semantic_end_id)
        prompt_acoustics = pad_2D(prompt_acoustics, self.prompt_acoustic_eos)
        target_acoustics = pad_2D(target_acoustics, self.target_acoustic_eos)
        # print('target_acoustics ', target_acoustics)
        x_mask = (target_acoustics == self.target_acoustic_eos)
        # print('mask ', x_mask.shape)
        # assert 1==2
        new_samples = {}
        new_samples['prompt_semantics'] = torch.from_numpy(prompt_semantics)
        new_samples['target_semantics'] = torch.from_numpy(target_semantics)
        new_samples['prompt_acoustics'] = torch.from_numpy(prompt_acoustics)
        new_samples['target_acoustics'] = torch.from_numpy(target_acoustics)
        new_samples['x_mask'] = torch.from_numpy(x_mask[:, 0, :])
        return new_samples
