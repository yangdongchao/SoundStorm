# Use AudioTag tool BEATs to filter out audios who's top1 tag is not 'speech'
# non_speech.npy, 存储一个 python dict 表示非 speech 类型的音频的 tag, 更小，加载和搜索速度更快
# audio_tag 目录存储 {utt_id}.txt, 第一行是小写的 top1 tag 
import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch
import tqdm
from soundstorm.s1.AR.exps.beats.BEATs import BEATs
from soundstorm.s1.AR.exps.beats.BEATs import BEATsConfig
from soundstorm.s1.AR.exps.beats.config import id_name_dict
from soundstorm.s2.exps.hubert.feature_utils import get_shard_range
from soundstorm.utils import check_txt_file


def get_BEATs_top1(wav,
                   BEATs_model,
                   BEATs_label_dict,
                   device: str='cpu',
                   topk: int=1):
    wav = torch.tensor(wav).unsqueeze(0).to(device)
    padding_mask = torch.zeros(wav.shape).bool().to(device)
    probs = BEATs_model.extract_features(wav, padding_mask=padding_mask)[0]
    # 单条推理
    probs = probs[0]
    topk_label_prob, topk_label_idx = probs.topk(k=topk)
    topk_label = [
        BEATs_label_dict[label_idx.item()] for label_idx in topk_label_idx
    ]
    topk_label_name = [id_name_dict[label] for label in topk_label]
    top1_label = topk_label_name[0]
    return top1_label


def process_sentence(args,
                     fp: Path,
                     train_dump_dir: Path,
                     dev_dump_dir: Path,
                     test_dump_dir: Path,
                     VAD_dict,
                     BEATs_model,
                     BEATs_label_dict,
                     device: str='cpu'):
    utt_id = fp.stem
    sr = args.sr
    record = []
    train_audio_tag_dir = train_dump_dir / "audio_tag"
    train_audio_tag_dir.mkdir(parents=True, exist_ok=True)

    dev_audio_tag_dir = dev_dump_dir / "audio_tag"
    dev_audio_tag_dir.mkdir(parents=True, exist_ok=True)

    test_audio_tag_dir = test_dump_dir / "audio_tag"
    test_audio_tag_dir.mkdir(parents=True, exist_ok=True)

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
                audio_tag_path = test_audio_tag_dir / (split_name + ".txt")
            elif index == len_dict - 2:
                subset = 'dev'
                audio_tag_path = dev_audio_tag_dir / (split_name + ".txt")
            else:
                subset = 'train'
                audio_tag_path = train_audio_tag_dir / (split_name + ".txt")

            if os.path.exists(audio_tag_path) and check_txt_file(
                    audio_tag_path):
                # print(audio_tag_path, 'exits!')
                pass
            else:
                # 这里加判断保证在 sub wav 的循环中只 load 一次
                if wav is None:
                    # load big wav
                    # 在最外层 load 如果 sub wav 的特征都存在了就会白白消耗 load 的时间
                    wav, _ = librosa.load(str(fp), sr=sr)
                sub_wav = wav[int(start * sr):int(end * sr)]
                audio_tag_top1 = get_BEATs_top1(
                    wav=sub_wav,
                    BEATs_model=BEATs_model,
                    BEATs_label_dict=BEATs_label_dict,
                    device=device)

                with open(audio_tag_path, 'w') as f:
                    f.write(audio_tag_top1)

            sub_record = {
                "utt_id": split_name,
                "audio_tag_path": audio_tag_path,
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
                      BEATs_model,
                      BEATs_label_dict,
                      device: str='cpu',
                      nprocs: int=1):
    print("nprocs:", nprocs)
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
                BEATs_model=BEATs_model,
                BEATs_label_dict=BEATs_label_dict,
                device=device)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, args, fp,
                                         train_dump_dir, dev_dump_dir,
                                         test_dump_dir, VAD_dict, BEATs_model,
                                         BEATs_label_dict, device)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    # torch.save() to a large `.pth` file
    non_speech_dict = dict()
    non_speech_dict['train'] = {}
    non_speech_dict['dev'] = {}
    non_speech_dict['test'] = {}
    # record 是 List of Dict, 一条大 wav 一个 record，一条小 wav 一个 sub_recored
    print(f"start to save {args.rank}_{args.nshard}.npy ...")
    save_start_time = time.time()
    for record in tqdm.tqdm(results, total=len(results), colour='green'):
        for sub_record in record:
            # 这里加 try, 因为 txt 文件可能损坏
            try:
                utt_id = sub_record["utt_id"]
                subset = sub_record["subset"]
                audio_tag_top1 = check_txt_file(sub_record["audio_tag_path"])
                if audio_tag_top1 is not False:
                    if 'speech' not in audio_tag_top1.lower():
                        non_speech_dict[subset][utt_id] = audio_tag_top1
                    else:
                        # print(f'audio tag result of {utt_id} is speech')
                        pass
                else:
                    print(f'audio tag result of {utt_id} is False')
            except Exception:
                print(f"{utt_id} occur Exception")
                traceback.print_exc()
                continue

    train_filename = train_dump_dir / f'non_speech_{args.rank}_{args.nshard}.npy'
    dev_filename = dev_dump_dir / f'non_speech_{args.rank}_{args.nshard}.npy'
    test_filename = test_dump_dir / f'non_speech_{args.rank}_{args.nshard}.npy'
    np.save(train_filename, non_speech_dict['train'])
    print(f"npy file '{train_filename}' write down")

    np.save(dev_filename, non_speech_dict['dev'])
    print(f"npy file '{dev_filename}' write down")

    np.save(test_filename, non_speech_dict['test'])
    print(f"npy file '{test_filename}' write down")
    print('time of save stage:', time.time() - save_start_time)


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Use AudioTag tool BEATs to filter out audios who's top1 tag is not 'speech'."
    )

    parser.add_argument(
        "--data_dir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="directory to dump feature files.")

    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

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
    parser.add_argument("--nshard", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)

    # for BEATs
    parser.add_argument(
        "--BEATs_ckpt_path",
        type=str,
        default='./pretrained_model/BEATs_iter1_finetuned_on_AS2M_cpt1.pt')

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

    BEATs_ckpt = torch.load(args.BEATs_ckpt_path)

    BEATs_cfg = BEATsConfig(BEATs_ckpt['cfg'])
    BEATs_model = BEATs(BEATs_cfg)
    BEATs_model.load_state_dict(BEATs_ckpt['model'])
    BEATs_model.eval()
    # cpu or cuda
    device = 'cpu'
    BEATs_model.to(device)

    BEATs_label_dict = BEATs_ckpt['label_dict']

    # 每条大 wav 分出一个 dev 一个 test，比例大概是 96:2:2
    if all_wav_files:
        process_sentences(
            args=args,
            fps=all_wav_files,
            train_dump_dir=train_dump_dir,
            dev_dump_dir=dev_dump_dir,
            test_dump_dir=test_dump_dir,
            VAD_dict=VAD_dict,
            BEATs_model=BEATs_model,
            BEATs_label_dict=BEATs_label_dict,
            device=device,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
