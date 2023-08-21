"""
1. read text of dataset
2. text -> IPA by GruutPhonemizer
3. save out a *.npy dict for all text
my_dict = {"utt_id1": text1, "utt_id2": text2}
np.save(output_filename, my_dict)
my_dict = np.load(output_filename, allow_pickle=True).item()
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from pathlib import Path
from typing import List

import numpy as np
import tqdm
from soundstorm.s1.AR.text_processing.phonemizer import GruutPhonemizer


def read_txt(txt_file):
    utt_name = txt_file.stem
    utt_id = utt_name.split('.')[0]
    try:
        with open(txt_file, 'r') as file:
            txt = file.readline()
        record = {"utt_id": utt_id, "txt": txt}
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        return None
    return record


def read_txts(txt_files: List[Path], nprocs: int=1):
    if nprocs == 1:
        results = []
        for txt_file in tqdm.tqdm(txt_files, total=len(txt_files)):
            record = read_txt(txt_file=txt_file)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(txt_files)) as progress:
                for txt_file in txt_files:
                    future = pool.submit(read_txt, txt_file)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    results.sort(key=itemgetter("utt_id"))
    return_list = []
    for item in results:
        return_list.append((item["utt_id"], item["txt"]))
    return return_list


def process_sentence(item, phonemizer):
    utt_id, text = item
    try:
        phonemes = phonemizer.phonemize(text, espeak=False)
        record = {"utt_id": utt_id, "phonemes": phonemes}
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        return None
    return record


def process_sentences(items, phonemizer, output_dir, nprocs: int=1):
    if nprocs == 1:
        results = []
        for item in tqdm.tqdm(items, total=len(items)):
            record = process_sentence(item=item, phonemizer=phonemizer)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(items)) as progress:
                for item in items:
                    future = pool.submit(process_sentence, item, phonemizer)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)
    results.sort(key=itemgetter("utt_id"))
    npy_dict = {}
    for item in results:
        utt_id = item["utt_id"]
        phonemes = item["phonemes"]
        npy_dict[utt_id] = phonemes
    filename = output_dir / 'phonemes.npy'
    np.save(filename, npy_dict)
    print(f"npy file '{filename}' write down")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(description="Get phones for datasets")

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
        "--num-cpu", type=int, default=1, help="number of process.")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    # use absolute path
    dump_dir = dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    assert data_dir.is_dir()

    if args.dataset == "ljspeech":
        data_dict = {}
        text_path = data_dir / 'metadata.csv'
        with open(text_path, 'r') as rf:
            for line in rf:
                line_list = line.strip().split('|')
                utt_id = line_list[0]
                raw_text = line_list[-1]
                data_dict[utt_id] = raw_text

        sorted_dict = sorted(data_dict.items())

        num_train = 12900
        num_dev = 100
        # (utt_id, txt)
        train_txts = sorted_dict[:num_train]
        dev_txts = sorted_dict[num_train:num_train + num_dev]
        test_txts = sorted_dict[num_train + num_dev:]

    elif args.dataset == "libritts":
        '''
        we use train-clean-100、train-clean-360、train-other-500 here 
        and split dev and test from them, don't use test-* and dev-* cause the speakers are disjoint
        the file structure is LibriTTS_R/train-clean-100/spkid/*/*.wav
        there are about 2311 in these subsets, we split 1 dev and 1 test wav out from each speaker
        '''
        txt_files = []
        train_txt_files = []
        dev_txt_files = []
        test_txt_files = []
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
                txt_files = sorted(
                    list((sub_dataset_dir / speaker).rglob(
                        "*/*.normalized.txt")))
                # filter out ._*.wav
                txt_files = [
                    file for file in txt_files if not file.name.startswith('._')
                ]
                train_txt_files += txt_files[:-sub_num_dev * 2]
                dev_txt_files += txt_files[-sub_num_dev * 2:-sub_num_dev]
                test_txt_files += txt_files[-sub_num_dev:]
        print("len(train_txt_files):", len(train_txt_files))
        print("len(dev_txt_files):", len(dev_txt_files))
        print("len(test_txt_files):", len(test_txt_files))

        train_txts = read_txts(train_txt_files)
        dev_txts = read_txts(dev_txt_files)
        test_txts = read_txts(test_txt_files)

    else:
        print("dataset should in {ljspeech, libritts} now!")

    train_dump_dir = dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dump_dir / "test"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    phonemizer = GruutPhonemizer(language='en-us')

    # process for the 3 sections
    if train_txts:
        process_sentences(
            items=train_txts,
            output_dir=train_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)
    if dev_txts:
        process_sentences(
            items=dev_txts,
            output_dir=dev_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)
    if test_txts:
        process_sentences(
            items=test_txts,
            output_dir=test_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
