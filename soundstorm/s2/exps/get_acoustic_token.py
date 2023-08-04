import argparse
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
import tqdm
from academicodec.models.encodec.net3 import SoundStream
from academicodec.models.encodec.test import remove_encodec_weight_norm
from academicodec.models.hificodec.vqvae import VQVAE


def process_sentence(args, fp: Path, output_dir: Path, codec_extractor):
    utt_id = fp.stem
    # for vctk
    if utt_id.endswith("_mic2"):
        utt_id = utt_id[:-5]
    record = None
    acoustic_token_dir = output_dir / "acoustic_token" / args.codec_name
    acoustic_token_dir.mkdir(parents=True, exist_ok=True)
    try:
        acoustic_token_path = acoustic_token_dir / (utt_id + ".npy")
        if os.path.exists(acoustic_token_path):
            # print(acoustic_token_path, 'exits!')
            pass
        else:
            # wav.shape (T, )
            wav, _ = librosa.load(str(fp), sr=args.sr)
            # wav.shape (1, T)
            wav = torch.tensor(wav).unsqueeze(0)
            wav = wav.cuda()
            if args.codec_name == 'hificodec':
                # (1, T, 4)
                acoustic_token = codec_extractor.encode(wav)
                # trans acoustic_token.shape to (Nq, T)
                acoustic_token = acoustic_token.squeeze(0).transpose(0, 1)
            elif args.codec_name == 'encodec':
                # wav.shape (1, 1, T)
                wav = wav.unsqueeze(1)
                # (24, 1, T)
                acoustic_token = codec_extractor.encode(
                    wav, target_bw=args.target_bw)
                # trans acoustic_token.shape to (Nq, T)
                acoustic_token = acoustic_token.squeeze(1)
            else:
                print("Please input the right codec_name!")

            acoustic_token_np = acoustic_token.detach().cpu().numpy()
            np.save(acoustic_token_path, acoustic_token_np)
        record = {"utt_id": utt_id, "acoustic_token_path": acoustic_token_path}
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        return None

    return record


def process_sentences(args,
                      fps: List[Path],
                      output_dir: Path,
                      codec_extractor,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                args=args,
                fp=fp,
                output_dir=output_dir,
                codec_extractor=codec_extractor)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, args, fp, output_dir,
                                         codec_extractor)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    results.sort(key=itemgetter("utt_id"))
    # torch.save() to a large `.pth` file
    acoustic_token_dict = {}
    for item in results:
        utt_id = item["utt_id"]
        acoustic_token_np = np.load(item["acoustic_token_path"])
        acoustic_token = torch.tensor(acoustic_token_np)
        acoustic_token_dict[utt_id] = acoustic_token
    filename = output_dir / "acoustic_token" / (args.codec_name + '.pth')
    torch.save(acoustic_token_dict, filename)
    print(f"pth file '{filename}' write down")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--codec_name",
        default="hificodec",
        type=str,
        help="name of codec, should in {hificodec, encodec} now")

    parser.add_argument(
        "--dataset",
        default="ljspeech",
        type=str,
        help="name of dataset, should in {ljspeech, libritts} now")

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

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    # use absolute path
    dump_dir = dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    assert data_dir.is_dir()

    if args.dataset == "ljspeech":
        wav_files = sorted(list((data_dir / "wavs").rglob("*.wav")))
        # split data into 3 sections
        num_train = 12900
        num_dev = 100
        train_wav_files = wav_files[:num_train]
        dev_wav_files = wav_files[num_train:num_train + num_dev]
        test_wav_files = wav_files[num_train + num_dev:]
    elif args.dataset == "libritts":
        '''
        we use train-clean-100、train-clean-360、train-other-500 here 
        and split dev and test from them, don't use test-* and dev-* cause the speakers are disjoint
        the file structure is LibriTTS_R/train-clean-100/spkid/*/*.wav
        there are about 2311 in these subsets, we split 1 dev and 1 test wav out from each speaker
        '''
        train_wav_files = []
        dev_wav_files = []
        test_wav_files = []
        sub_num_dev = 1
        for sub_dataset_name in {
                "train-clean-100", "train-clean-360", "train-other-500"
        }:
            sub_dataset_dir = data_dir / sub_dataset_name
            # filter out hidden files
            speaker_list = [
                file for file in os.listdir(sub_dataset_dir)
                if not file.startswith('.')
            ]
            for speaker in speaker_list:
                wav_files = sorted(list((sub_dataset_dir / speaker).rglob("*/*.wav")))
                # filter out ._*.wav
                wav_files = [file for file in wav_files if not file.name.startswith('._')]
                train_wav_files += wav_files[:-sub_num_dev * 2]
                dev_wav_files += wav_files[-sub_num_dev * 2:-sub_num_dev]
                test_wav_files += wav_files[-sub_num_dev:]
        print("len(train_wav_files):", len(train_wav_files))
        print("len(dev_wav_files):", len(dev_wav_files))
        print("len(test_wav_files):", len(test_wav_files))

    elif args.dataset == "librilight":
        print("not ready yet.")

    else:
        print("dataset should in {ljspeech, libritts} now!")

    train_dump_dir = dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dump_dir / "test"
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

    # process for the 3 sections
    if train_wav_files:
        process_sentences(
            args=args,
            fps=train_wav_files,
            output_dir=train_dump_dir,
            codec_extractor=codec_extractor,
            nprocs=args.num_cpu)
    if dev_wav_files:
        process_sentences(
            args=args,
            fps=dev_wav_files,
            output_dir=dev_dump_dir,
            codec_extractor=codec_extractor,
            nprocs=args.num_cpu)
    if test_wav_files:
        process_sentences(
            args=args,
            fps=test_wav_files,
            output_dir=test_dump_dir,
            codec_extractor=codec_extractor,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
