import os
import re

import numpy as np


def str2bool(str):
    return True if str.lower() == 'true' else False


def get_newest_ckpt(string_list):
    # 定义一个正则表达式模式，用于匹配字符串中的数字
    pattern = r'epoch=(\d+)-step=(\d+)\.ckpt'

    # 使用正则表达式提取每个字符串中的数字信息，并创建一个包含元组的列表
    extracted_info = []
    for string in string_list:
        match = re.match(pattern, string)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            extracted_info.append((epoch, step, string))
    # 按照 epoch 后面的数字和 step 后面的数字进行排序
    sorted_info = sorted(
        extracted_info, key=lambda x: (x[0], x[1]), reverse=True)
    # 获取最新的 ckpt 文件名
    newest_ckpt = sorted_info[0][2]
    return newest_ckpt


def get_files_by_suffix(path: str, suffix: str):
    files = []
    for file_name in os.listdir(path):
        if file_name.endswith(suffix):
            files.append(os.path.join(path, file_name))
    return files


# 增加 prefix 是因为 phonemes_*.npy 同目录下还有 txt_*.npy
def get_files_by_prefix_suffix(path: str, prefix: str, suffix: str):
    files = []
    for file_name in os.listdir(path):
        if file_name.startswith(prefix) and file_name.endswith(suffix):
            files.append(os.path.join(path, file_name))
    return files


# 文本存在且不为空时 return True
def check_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.readline().strip()
        assert text.strip() != ''
        return text
    except Exception:
        return False
    return False


def check_numpy_file(file_path):
    try:
        # 尝试加载 numpy 文件
        np.load(file_path)
        # print("文件存在且没有损坏。")
        return True
    except Exception:
        # traceback.print_exc()
        print(f'Cannot load {file_path}, will return False and regenerate it')
        return False
    return False
