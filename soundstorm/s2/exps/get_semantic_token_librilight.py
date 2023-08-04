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
from soundstorm.s2.models.hubert.semantic_tokenizer import SemanticTokenizer

# ThreadPoolExecutor 适用于 I/O 密集型任务，具有轻量级线程切换的优势
# ProcessPoolExecutor 适用于 CPU 密集型任务，可以充分利用多核处理器的优势


def process_sentence(fp: Path, output_dir: Path, semantic_tokenizer):
    utt_id = fp.stem
    # for vctk
    if utt_id.endswith("_mic2"):
        utt_id = utt_id[:-5]
    record = None
    semantic_token_dir = output_dir / "semantic_token"
    semantic_token_dir.mkdir(parents=True, exist_ok=True)
    try:
        semantic_token_path = semantic_token_dir / (utt_id + ".npy")
        if os.path.exists(semantic_token_path):
            # print(semantic_token_path, 'exits!')
            pass
        else:
            # reading, resampling may occur
            # mHuBERT's sr = 16000
            wav, _ = librosa.load(str(fp), sr=16000)
            wav = torch.tensor(wav).unsqueeze(0)
            semantic_token = semantic_tokenizer.tokenize(wav)
            semantic_token_np = semantic_token.detach().cpu().numpy()
            np.save(semantic_token_path, semantic_token_np)
        record = {"utt_id": utt_id, "semantic_token_path": semantic_token_path}
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        return None
    return record


def process_sentences(fps: List[Path],
                      output_dir: Path,
                      semantic_tokenizer,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                fp=fp,
                output_dir=output_dir,
                semantic_tokenizer=semantic_tokenizer)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, fp, output_dir,
                                         semantic_tokenizer)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    data = [['item_name', 'semantic_audio']]
    results.sort(key=itemgetter("utt_id"))
    for item in results:
        utt_id = item["utt_id"]
        # old hubert_kmeans shape is (T,), new hubert_kmeans shape is (1, T)
        # so add [0] here
        semantic_token = np.load(item["semantic_token_path"])[0].tolist()
        semantic_token_str = ' '.join(str(x) for x in semantic_token)
        data.append([utt_id, semantic_token_str])
    delimiter = '\t'
    filename = output_dir / "semantic_token.tsv"
    with open(filename, 'w', encoding='utf-8') as writer:
        for row in data:
            line = delimiter.join(row)  # 使用制表符拼接每行数据
            writer.write(line + '\n')

    print(f"tsv file '{filename}' write down")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

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
        wav_files = []
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
                wav_files = sorted(
                    list((sub_dataset_dir / speaker).rglob("*/*.wav")))
                # filter out ._*.wav
                wav_files = [
                    file for file in wav_files if not file.name.startswith('._')
                ]
                train_wav_files += wav_files[:-sub_num_dev * 2]
                dev_wav_files += wav_files[-sub_num_dev * 2:-sub_num_dev]
                test_wav_files += wav_files[-sub_num_dev:]
        print("len(train_wav_files):", len(train_wav_files))
        print("len(dev_wav_files):", len(dev_wav_files))
        print("len(test_wav_files):", len(test_wav_files))

    else:
        print("dataset should in {ljspeech, libritts} now!")

    train_dump_dir = dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dump_dir / "test"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    print("args.layer:", args.layer)

    semantic_tokenizer = SemanticTokenizer(
        hubert_path=args.hubert_path,
        quantizer_path=args.quantizer_path,
        duplicate=True,
        output_layer=args.layer)

    # process for the 3 sections
    if train_wav_files:
        process_sentences(
            fps=train_wav_files,
            output_dir=train_dump_dir,
            semantic_tokenizer=semantic_tokenizer,
            nprocs=args.num_cpu)
    if dev_wav_files:
        process_sentences(
            fps=dev_wav_files,
            output_dir=dev_dump_dir,
            semantic_tokenizer=semantic_tokenizer,
            nprocs=args.num_cpu)
    if test_wav_files:
        process_sentences(
            fps=test_wav_files,
            output_dir=test_dump_dir,
            semantic_tokenizer=semantic_tokenizer,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
