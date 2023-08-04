# modified by https://github.com/facebookresearch/libri-light/blob/main/data_preparation/cut_by_vad.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''
1. 短的 vad 合并成 25s, 保留多个 vad 之间的静音
2. 若大于 50s 则切成 n 个 25s 的
3. 25s 到 50s 保留
这样会保证最终的音频在 0 ~ 50s 都用
VAD 时长分布参考 https://github.com/yt605155624/mine_images/issues/1#issuecomment-1661367337
子音频️以 **==`{spk_id}_{wav_name}_{index:04}`==** 作为唯一标识
'''
import argparse
import json
import multiprocessing
import pathlib

import tqdm


def save(seq, fname, index, extension):
    fname.parent.mkdir(exist_ok=True, parents=True)
    # canterburytales_18_chaucer_64kb -> canterburytales_18_chaucer_64kb_0000
    # 也可以接收 > 10000 的 index
    file_name = fname.parent / (fname.stem + f"_{index:04}{extension}")
    with open(file_name, 'w', encoding='utf-8') as writer:
        for line in seq:
            writer.write(str(line) + '\n')


def cut_sequence(vad, path_out, target_len_sec, out_extension):
    to_stitch = []
    length_accumulated = 0.0
    # VAD merge 相关代码主要编辑这里

    i = 0
    for start, end in vad:
        slice = end - start
        to_stitch.append(slice)

    if to_stitch:
        save(to_stitch, path_out, i, out_extension)


def cut_book(task):
    path_book, root_out, target_len_sec, extension = task

    speaker = pathlib.Path(path_book.parent.name)
    # 每个 book 下可能有多个长音频
    for i, meta_file_path in enumerate(path_book.glob('*.json')):
        with open(meta_file_path, 'r') as f:
            meta = json.loads(f.read())
        book_id = meta['book_meta']['id']
        vad = meta['voice_activity']
        # small/151/canterburytales_librivox_64kb_mp3 -> small/15/342
        # fengyufei: LibriLight 每个音频文件（*.flac）在整个数据集里面是唯一的
        # 所以 vad 信息可以用一个字典唯一标识, 可以先多进程写小文件再一次性 merge
        path_out = root_out / speaker / book_id / (meta_file_path.stem)
        cut_sequence(vad, path_out, target_len_sec, extension)


def cut(input_dir,
        output_dir,
        target_len_sec=30,
        n_process=32,
        out_extension='.flac'):
    # book 级别的目录
    list_dir = pathlib.Path(input_dir).glob('*/*')
    list_dir = [x for x in list_dir if x.is_dir()]
    
    print(f"{len(list_dir)} directories detected")
    print(f"Launching {n_process} processes")

    tasks = [(path_book, output_dir, target_len_sec, out_extension)
             for path_book in list_dir]
    
    with multiprocessing.Pool(processes=n_process) as pool:
        for _ in tqdm.tqdm(
                pool.imap_unordered(cut_book, tasks), total=len(tasks)):
            pass


def parse_args():

    parser = argparse.ArgumentParser(description="Cut a dataset in small "
                                     "sequences using VAD files")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/home/yuantian04/datasets/LibriLight/small',
        help="Path to the input directory",
        required=True)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/yuantian04/datasets/LibriLight_VAD/small',
        help="Path to the output directory",
        required=True)

    parser.add_argument(
        '--target_len_sec',
        type=int,
        default=60,
        help="Target time, in seconds of each output sequence"
        "(default is 60)")
    parser.add_argument(
        '--n_workers',
        type=int,
        default=32,
        help="Number of parallel worker processes")
    parser.add_argument(
        '--out_extension', type=str, default=".txt", help="Output extension")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    cut(args.input_dir, args.output_dir, args.target_len_sec, args.n_workers,
        args.out_extension)
