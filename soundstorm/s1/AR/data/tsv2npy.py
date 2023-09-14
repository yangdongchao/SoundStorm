# semantic_token 存成 tsv，在 dataloader 中需要转换成 list(int) 非常占用内存，所以直接转成 npy 文件存 list(int)
import os

import numpy as np
import pandas as pd

# 指定要处理的根目录
root_directory = '/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_librilight/large'  # 将此路径替换为你的根目录路径


def convert_tsv_to_dict(semantic_file):
    semantic_data = pd.read_csv(semantic_file, delimiter='\t')
    semantic_data_len = len(semantic_data)
    semantic_dict = {}
    for i in range(semantic_data_len):
        # 先依次遍历
        # get str
        item_name = semantic_data['item_name'][i]
        semantic_str = semantic_data['semantic_audio'][i]
        # get token list
        semantic_ids = [int(idx) for idx in semantic_str.split(' ')]
        semantic_dict[item_name] = semantic_ids
    return semantic_dict


# 遍历根目录以及其所有子目录
for dirname in os.listdir(root_directory):
    if dirname in {'dev', 'test', 'train'}:
        dirpath = os.path.join(root_directory, dirname)
        if os.path.isdir(dirpath):
            for filename in os.listdir(dirpath):
                if filename.endswith('.tsv'):
                    # 构建完整的输入文件路径
                    input_file_path = os.path.join(dirpath, filename)
                    print("input_file_path:", input_file_path)
                    # 构建输出文件的路径，将后缀名从 .tsv 更改为 .npy
                    output_file_path = os.path.splitext(input_file_path)[
                        0] + '.npy'
                    semantic_dict = convert_tsv_to_dict(input_file_path)
                    print("output_file_path:", output_file_path)
                    # # 读取 .tsv 文件并将其保存为 .npy 文件
                    np.save(output_file_path, semantic_dict)
