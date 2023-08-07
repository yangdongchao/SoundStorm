import argparse
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch
import tqdm
import time
from soundstorm.s2.exps.hubert.feature_utils import get_shard_range
from soundstorm.s2.models.hubert.semantic_tokenizer import SemanticTokenizer

# ThreadPoolExecutor 适用于 I/O 密集型任务，具有轻量级线程切换的优势
# ProcessPoolExecutor 适用于 CPU 密集型任务，可以充分利用多核处理器的优势


def process_sentence(args,
                     fp: Path,
                     train_dump_dir: Path,
                     dev_dump_dir: Path,
                     test_dump_dir: Path,
                     VAD_dict,
                     semantic_tokenizer):
    utt_id = fp.stem
    sr = args.sr
    record = []
    train_semantic_token_dir = train_dump_dir / "semantic_token"
    train_semantic_token_dir.mkdir(parents=True, exist_ok=True)

    dev_semantic_token_dir = dev_dump_dir / "semantic_token"
    dev_semantic_token_dir.mkdir(parents=True, exist_ok=True)

    test_semantic_token_dir = test_dump_dir / "semantic_token"
    test_semantic_token_dir.mkdir(parents=True, exist_ok=True)

    try:
        # get info for path
        wav_path_list = str(fp).strip().split('/')
        sub_dataset, spk_id, book_name = wav_path_list[-4], wav_path_list[
            -3], wav_path_list[-2]
        wav_name = wav_path_list[-1][:-5]
        assert wav_name == utt_id
        # key_name for big wav
        key_name = f'{wav_name}#{sub_dataset}#{spk_id}#{book_name}'
        # 判断 VAD 字典中不存在该条音频信息的情况
        if key_name not in VAD_dict.keys():
            print(key_name, 'not in VAD_dict !')
            return record
        wav = None
        sorted_split_VAD_dict = sorted(VAD_dict[key_name].items())
        len_dict = len(sorted_split_VAD_dict)
        for index, item in enumerate(sorted_split_VAD_dict):
            split_name, value = item
            start, end = value
            # train | dev | test
            if index == len_dict - 1:
                subset = 'test'
                semantic_token_path = test_semantic_token_dir / (
                    split_name + ".npy")
            elif index == len_dict - 2:
                subset = 'dev'
                semantic_token_path = dev_semantic_token_dir / (
                    split_name + ".npy")
            else:
                subset = 'train'
                semantic_token_path = train_semantic_token_dir / (
                    split_name + ".npy")

            if os.path.exists(semantic_token_path):
                # print(semantic_token_path, 'exits!')
                pass
            else:
                # 这里加判断保证在 sub wav 的循环中只 load 一次
                if wav is None:
                    # load big wav
                    # 在最外层 load 如果 sub wav 的特征都存在了就会白白消耗 load 的时间
                    wav, _ = librosa.load(str(fp), sr=sr)
                sub_wav = wav[int(start * sr):int(end * sr)]
                sub_wav = torch.tensor(sub_wav).unsqueeze(0)
                semantic_token = semantic_tokenizer.tokenize(sub_wav)
                semantic_token_np = semantic_token.detach().cpu().numpy()
                np.save(semantic_token_path, semantic_token_np)
            sub_record = {
                "utt_id": split_name,
                "semantic_token_path": semantic_token_path,
                "subset": subset
            }
            # recodrd 变成 List of Dict
            record.append(sub_record)
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        # record 有可能是一个不完整的 List
        return record
    return record


def process_sentences(args,
                      fps: Path,
                      train_dump_dir: Path,
                      dev_dump_dir: Path,
                      test_dump_dir: Path,
                      VAD_dict,
                      semantic_tokenizer,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                args=args,
                fp=fp,
                train_dump_dir=train_dump_dir,
                dev_dump_dir=dev_dump_dir,
                test_dump_dir=test_dump_dir,
                VAD_dict=VAD_dict,
                semantic_tokenizer=semantic_tokenizer)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, args, fp,
                                         train_dump_dir, dev_dump_dir,
                                         test_dump_dir, VAD_dict,
                                         semantic_tokenizer)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    train_data = [['item_name', 'semantic_audio']]
    dev_data = [['item_name', 'semantic_audio']]
    test_data = [['item_name', 'semantic_audio']]

    # record 是 List of Dict, 一条大 wav 一个 record，一条小 wav 一个 sub_recored
    for record in results:
        for sub_record in record:
            try:
                utt_id = sub_record["utt_id"]
                subset = sub_record["subset"]
                # old hubert_kmeans shape is (T,), new hubert_kmeans shape is (1, T)
                # so add [0] here
                semantic_token = np.load(
                    sub_record["semantic_token_path"])[0].tolist()
                semantic_token_str = ' '.join(str(x) for x in semantic_token)
                if subset == "train":
                    train_data.append([utt_id, semantic_token_str])
                elif subset == "dev":
                    dev_data.append([utt_id, semantic_token_str])
                # test
                else:
                    test_data.append([utt_id, semantic_token_str])
            except Exception:
                print(f"{utt_id} occur Exception")
                traceback.print_exc()
                continue

    delimiter = '\t'
    train_filename = train_dump_dir / f'semantic_token_{args.rank}_{args.nshard}.tsv'
    dev_filename = dev_dump_dir / f'semantic_token_{args.rank}_{args.nshard}.tsv'
    test_filename = test_dump_dir / f'semantic_token_{args.rank}_{args.nshard}.tsv'
    print(f"start to save {args.rank}_{args.nshard}.tsv ...")
    save_start_time = time.time()
    with open(train_filename, 'w', encoding='utf-8') as writer:
        for row in train_data:
            line = delimiter.join(row)
            writer.write(line + '\n')
    print(f"tsv file '{train_filename}' write down")

    with open(dev_filename, 'w', encoding='utf-8') as writer:
        for row in dev_data:
            line = delimiter.join(row)
            writer.write(line + '\n')
    print(f"tsv file '{dev_filename}' write down")

    with open(test_filename, 'w', encoding='utf-8') as writer:
        for row in test_data:
            line = delimiter.join(row)
            writer.write(line + '\n')
    print(f"tsv file '{test_filename}' write down")
    print('time of save stage:', time.time() - save_start_time)


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features for LibriLight.")

    parser.add_argument(
        "--data_dir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="directory to dump feature files.")

    parser.add_argument(
        "--hubert_path", type=str, default='./hubert_base_ls960.pt')

    parser.add_argument(
        "--quantizer_path",
        type=str,
        default='./hubert_base_ls960_L9_km500.bin')

    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    parser.add_argument(
        "--layer",
        type=int,
        default=10,
        help="use which layer of feature of hubert, should be same with it in exp/dump_hubert_feature.py"
    )
    parser.add_argument(
        '--sr', type=int, default=16000, help='sample rate of model')

    # For LibriLight dataset
    parser.add_argument(
        "--sub_dataset",
        default="small",
        type=str,
        help="name of sub dataset of LibriLight",
        choices=['small', 'medium', 'large', 'duplicate'], )
    parser.add_argument(
        "--VAD_path", type=str, default='./VAD/librilight_segment_dict.npy')
    parser.add_argument("--nshard", type=int, default=5)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    # use absolute path
    dump_dir = dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    assert data_dir.is_dir()

    # sub_dataset here
    sub_dataset_dir = data_dir / args.sub_dataset
    # olny spk_id in list, sort by lexicographical order 
    speaker_list = sorted(os.listdir(sub_dataset_dir))
    start, end = get_shard_range(len(speaker_list), args.nshard, args.rank)
    # speaker_list for this rank
    speaker_list = speaker_list[start:end]

    all_wav_files = []

    for speaker in speaker_list:
        wav_files = sorted(list((sub_dataset_dir / speaker).rglob("*/*.flac")))
        # filter out ._*.flac
        wav_files = [
            file for file in wav_files if not file.name.startswith('._')
        ]
        all_wav_files += wav_files

    print(f"num of wav files in rank {args.rank}:", len(all_wav_files))
    # get VAD info
    VAD_dict = np.load(args.VAD_path, allow_pickle=True).item()

    sub_dataset_dump_dir = dump_dir / args.sub_dataset
    sub_dataset_dump_dir.mkdir(parents=True, exist_ok=True)
    train_dump_dir = sub_dataset_dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = sub_dataset_dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = sub_dataset_dump_dir / "test"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    print("args.layer:", args.layer)

    semantic_tokenizer = SemanticTokenizer(
        hubert_path=args.hubert_path,
        quantizer_path=args.quantizer_path,
        duplicate=True,
        output_layer=args.layer)

    # 每条大 wav 分出一个 dev 一个 test，比例大概是 96:2:2
    if all_wav_files:
        process_sentences(
            args=args,
            fps=all_wav_files,
            train_dump_dir=train_dump_dir,
            dev_dump_dir=dev_dump_dir,
            test_dump_dir=test_dump_dir,
            VAD_dict=VAD_dict,
            semantic_tokenizer=semantic_tokenizer,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
