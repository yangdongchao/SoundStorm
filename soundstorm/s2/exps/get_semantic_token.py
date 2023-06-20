import argparse
import os
# ThreadPoolExecutor 适用于 I/O 密集型任务，具有轻量级线程切换的优势
# ProcessPoolExecutor 适用于 CPU 密集型任务，可以充分利用多核处理器的优势
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
import tqdm
from soundstorm.s2.models.mhubert.semantic_tokenizer import SemanticTokenizer


def process_sentence(fp: Path, output_dir: Path, semantic_tokenizer):
    utt_id = fp.stem
    # for vctk
    if utt_id.endswith("_mic2"):
        utt_id = utt_id[:-5]
    record = None
    semantic_token_dir = output_dir / "semantic_token"
    semantic_token_dir.mkdir(parents=True, exist_ok=True)
    # reading, resampling may occur
    # mHuBERT's sr = 16000
    try:
        semantic_token_path = semantic_token_dir / (utt_id + ".npy")
        if os.path.exists(semantic_token_path):
            # print(semantic_token_path, 'exits!')
            pass
        else:
            wav, _ = librosa.load(str(fp), sr=16000)
            wav = torch.tensor(wav).unsqueeze(0)
            semantic_token = semantic_tokenizer.tokenize(wav)
            semantic_token_np = semantic_token.detach().cpu().numpy()
            np.save(semantic_token_path, semantic_token_np)
        record = {"utt_id": utt_id, "semantic_token_path": semantic_token_path}
    except Exception:
        print("occur Exception")
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
        semantic_token = np.load(item["semantic_token_path"]).tolist()
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
        "--hubert_path", type=str, default='./mhubert_base_vp_en_es_fr_it3.pt')

    parser.add_argument(
        "--quantizer_path",
        type=str,
        default='./mhubert_base_vp_en_es_fr_it3_L11_km1000.bin')

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
        # 这里是 LJSpeech_mini 的，后续记得改
        num_train = 250
        num_dev = 10
        train_wav_files = wav_files[:num_train]
        dev_wav_files = wav_files[num_train:num_train + num_dev]
        test_wav_files = wav_files[num_train + num_dev:]
    elif args.dataset == "libritts":
        print("not ready yet.")

    else:
        print("dataset should in {ljspeech, libritts} now!")

    train_dump_dir = dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dump_dir / "test"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    semantic_tokenizer = SemanticTokenizer(
        hubert_path=args.hubert_path,
        quantizer_path=args.quantizer_path,
        duplicate=True)

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
