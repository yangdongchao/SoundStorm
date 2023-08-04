import argparse
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
import tqdm
from academicodec.models.encodec.net3 import SoundStream
from academicodec.models.encodec.test import remove_encodec_weight_norm
from academicodec.models.hificodec.vqvae import VQVAE
from soundstorm.s2.exps.hubert.feature_utils import get_shard_range


# 原本的泄愤，返回的是一个短条的信息
def process_sentence(args,
                     fp: Path,
                     train_dump_dir: Path,
                     dev_dump_dir: Path,
                     test_dump_dir: Path,
                     VAD_dict,
                     codec_extractor):
    utt_id = fp.stem
    sr = args.sr
    record = []

    train_acoustic_token_dir = train_dump_dir / "acoustic_token" / args.codec_name
    train_acoustic_token_dir.mkdir(parents=True, exist_ok=True)

    dev_acoustic_token_dir = dev_dump_dir / "acoustic_token" / args.codec_name
    dev_acoustic_token_dir.mkdir(parents=True, exist_ok=True)

    test_acoustic_token_dir = test_dump_dir / "acoustic_token" / args.codec_name
    test_acoustic_token_dir.mkdir(parents=True, exist_ok=True)

    try:
        # get info for path
        wav_path_list = str(fp).strip().split('/')
        sub_dataset, spk_id, book_name = wav_path_list[-4], wav_path_list[
            -3], wav_path_list[-2]
        wav_name = wav_path_list[-1][:-5]
        assert wav_name == utt_id
        # key_name for big wav
        key_name = f'{wav_name}#{sub_dataset}#{spk_id}#{book_name}'
        # load big wav
        wav, _ = librosa.load(str(fp), sr=sr)
        sorted_split_VAD_dict = sorted(VAD_dict[key_name].items())
        len_dict = len(sorted_split_VAD_dict)

        for index, item in enumerate(sorted_split_VAD_dict):
            split_name, value = item
            start, end = value
            sub_wav = wav[int(start * sr):int(end * sr)]
            # train | dev | test
            if index == len_dict - 1:
                subset = 'test'
                acoustic_token_path = test_acoustic_token_dir / (
                    split_name + ".npy")
            elif index == len_dict - 2:
                subset = 'dev'
                acoustic_token_path = dev_acoustic_token_dir / (
                    split_name + ".npy")
            else:
                subset = 'train'
                acoustic_token_path = train_acoustic_token_dir / (
                    split_name + ".npy")
            
            if os.path.exists(acoustic_token_path):
                # print(acoustic_token_path, 'exits!')
                pass
            else:
                # sub_wav.shape (1, T)
                sub_wav = torch.tensor(sub_wav).unsqueeze(0)
                sub_wav = sub_wav.cuda()
                if args.codec_name == 'hificodec':
                    # (1, T, 4)
                    acoustic_token = codec_extractor.encode(sub_wav)
                    # trans acoustic_token.shape to (Nq, T)
                    acoustic_token = acoustic_token.squeeze(0).transpose(0, 1)
                elif args.codec_name == 'encodec':
                    # sub_wav.shape (1, 1, T)
                    sub_wav = sub_wav.unsqueeze(1)
                    # (24, 1, T)
                    acoustic_token = codec_extractor.encode(
                        sub_wav, target_bw=args.target_bw)
                    # trans acoustic_token.shape to (Nq, T)
                    acoustic_token = acoustic_token.squeeze(1)
                else:
                    print("Please input the right codec_name!")

                acoustic_token_np = acoustic_token.detach().cpu().numpy()
                np.save(acoustic_token_path, acoustic_token_np)
            sub_record = {
                "utt_id": split_name,
                "acoustic_token_path": acoustic_token_path,
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
                      fps: List[Path],
                      train_dump_dir: Path,
                      dev_dump_dir: Path,
                      test_dump_dir: Path,
                      VAD_dict,
                      codec_extractor,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        # for fp in tqdm.tqdm(fps, total=len(fps)):
        for fp in fps:
            record = process_sentence(
                args=args,
                fp=fp,
                train_dump_dir=train_dump_dir,
                dev_dump_dir=dev_dump_dir,
                test_dump_dir=test_dump_dir,
                VAD_dict=VAD_dict,
                codec_extractor=codec_extractor)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(
                        process_sentence, args, fp, train_dump_dir,
                        dev_dump_dir, test_dump_dir, VAD_dict, codec_extractor)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)
    # torch.save() to a large `.pth` file
    acoustic_token_dict = dict()
    acoustic_token_dict['train'] = {}
    acoustic_token_dict['dev'] = {}
    acoustic_token_dict['test'] = {}
    # record 是 List of Dict, 一条大 wav 一个 record，一条小 wav 一个 sub_recored
    for record in results:
        for sub_record in record:
            utt_id = sub_record["utt_id"]
            subset = sub_record["subset"]
            acoustic_token_np = np.load(sub_record["acoustic_token_path"])
            acoustic_token = torch.tensor(acoustic_token_np)
            acoustic_token_dict[subset][utt_id] = acoustic_token
    train_filename = train_dump_dir / "acoustic_token" / f'{args.codec_name}_{args.rank}_{args.nshard}.pth'
    dev_filename = dev_dump_dir / "acoustic_token" / f'{args.codec_name}_{args.rank}_{args.nshard}.pth'
    test_filename = test_dump_dir / "acoustic_token" / f'{args.codec_name}_{args.rank}_{args.nshard}.pth'
    # print("acoustic_token_dict['train']",acoustic_token_dict['train'])
    torch.save(acoustic_token_dict['train'], train_filename)
    print(f"pth file '{train_filename}' write down")

    torch.save(acoustic_token_dict['dev'], dev_filename)
    print(f"pth file '{dev_filename}' write down")

    torch.save(acoustic_token_dict['train'], test_filename)
    print(f"pth file '{test_filename}' write down")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features for LibriLight.")

    parser.add_argument(
        "--codec_name",
        default="hificodec",
        type=str,
        help="name of codec, should in {hificodec, encodec} now")

    parser.add_argument(
        "--data_dir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="directory to dump feature files.")

    parser.add_argument(
        "--model_path", type=str, default='./HiFi-Codec-16k-320d')

    parser.add_argument(
        '--sr', type=int, default=16000, help='sample rate of model')

    # for HiFi-Codec
    parser.add_argument(
        "--config_path", type=str, default='./config_16k_320d.json')

    # for Encodec
    parser.add_argument(
        '--ratios',
        type=int,
        nargs='+',
        # probs(ratios) = hop_size, default for 16k_320d
        default=[8, 5, 4, 2],
        help='ratios of SoundStream / Encodec, shoud be set for different hop_size (32d, 320, 240d, ...)'
    )
    parser.add_argument(
        '--target_bandwidths',
        type=float,
        nargs='+',
        # default for 16k_320d
        default=[1, 1.5, 2, 4, 6, 12],
        help='target_bandwidths of net3.py')
    parser.add_argument(
        '--target_bw',
        type=float,
        # default for 16k_320d
        default=12,
        help='target_bw of net3.py')

    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

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

    print("len(all_wav_files):", len(all_wav_files))
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

    if args.codec_name == 'hificodec':
        model = VQVAE(
            config_path=args.config_path,
            ckpt_path=args.model_path,
            with_encoder=True)
        model.cuda()
        model.generator.remove_weight_norm()
        model.encoder.remove_weight_norm()
        model.eval()

    elif args.codec_name == 'encodec':
        model = SoundStream(
            n_filters=32,
            D=512,
            ratios=args.ratios,
            sample_rate=args.sr,
            target_bandwidths=args.target_bandwidths)
        parameter_dict = torch.load(args.model_path)
        new_state_dict = {}
        # k 为 module.xxx.weight, v 为权重
        for k, v in parameter_dict.items():
            # 截取 `module.` 后面的 xxx.weight
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.cuda()
        remove_encodec_weight_norm(model)
        model.eval()

    else:
        print("Please input the right codec_name!")

    codec_extractor = model

    # 每条大 wav 分出一个 dev 一个 test，比例大概是 96:2:2
    if all_wav_files:
        process_sentences(
            args=args,
            fps=all_wav_files,
            train_dump_dir=train_dump_dir,
            dev_dump_dir=dev_dump_dir,
            test_dump_dir=test_dump_dir,
            VAD_dict=VAD_dict,
            codec_extractor=codec_extractor,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
