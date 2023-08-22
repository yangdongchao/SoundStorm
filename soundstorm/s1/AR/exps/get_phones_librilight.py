"""
1. read text of dataset, for LibriLight read txt_*.npy -> 需要整理成 list(utt_id, txt) 的形式
2. text -> IPA by GruutPhonemizer
3. save out a *.npy dict for all text
4. LibriLight 每个 split 分开处理
my_dict = {"utt_id1": text1, "utt_id2": text2}
np.save(output_filename, my_dict)
my_dict = np.load(output_filename, allow_pickle=True).item()
"""
import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from pathlib import Path

import numpy as np
import tqdm
from soundstorm.s1.AR.text_processing.phonemizer import GruutPhonemizer
from soundstorm.utils import check_txt_file


def read_txts(txt_file: Path, nprocs: int=1):
    '''
    txt_file: path of npy dict, {"utt_id1": text1, "utt_id2": text2}
    '''
    txt_dict = np.load(txt_file, allow_pickle=True).item()
    #[(utt_id, txt), ...]
    return_list = list(txt_dict.items())
    return return_list


def process_sentence(item, phonemizer, output_dir):
    utt_id, text = item
    phonemes_dir = output_dir / "phonemes"
    phonemes_dir.mkdir(parents=True, exist_ok=True)
    phonemes_path = phonemes_dir / (utt_id + ".txt")
    try:
        if os.path.exists(phonemes_path) and check_txt_file(phonemes_path):
            # print(phonemes_path, 'exits!')
            pass
        else:
            phonemes = phonemizer.phonemize(text, espeak=False)
            with open(phonemes_path, 'w') as f:
                f.write(phonemes)
        record = {"utt_id": utt_id, "phonemes_path": phonemes_path}
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        return None
    return record


def process_sentences(args, items, phonemizer, output_dir, nprocs: int=1):
    print("nprocs:", nprocs)
    if nprocs == 1:
        results = []
        for item in tqdm.tqdm(items, total=len(items)):
            record = process_sentence(
                item=item, phonemizer=phonemizer, output_dir=output_dir)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(items)) as progress:
                for item in items:
                    future = pool.submit(process_sentence, item, phonemizer,
                                         output_dir)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    results.sort(key=itemgetter("utt_id"))

    npy_dict = {}
    print(f"start to save {args.rank}_{args.nshard}.npy ...")
    save_start_time = time.time()
    for item in tqdm.tqdm(results, total=len(results), colour='green'):
        # 这里加 try, 因为 txt 文件可能损坏
        try:
            utt_id = item["utt_id"]
            phonemes = check_txt_file(item["phonemes_path"])
            if phonemes is not False:
                npy_dict[utt_id] = phonemes
            else:
                print(f'phonemes of {utt_id} is False')
        except Exception:
            print(f"{utt_id} occur Exception")
            traceback.print_exc()
            continue

    filename = output_dir / f'phonemes_{args.rank}_{args.nshard}.npy'
    np.save(filename, npy_dict)
    print(f"npy file '{filename}' write down")
    print('time of save stage:', time.time() - save_start_time)


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Get phones for LibriLight dataset from txt_*.npy")

    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="directory to dump feature files.")
    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    parser.add_argument(
        '--train_txt_dir',
        type=str,
        default='dump/small/train/',
        help='dir of train txt files')
    parser.add_argument(
        '--dev_txt_dir',
        type=str,
        default='dump/small/dev/',
        help='dir of dev txt files')
    parser.add_argument(
        '--test_txt_dir',
        type=str,
        default='dump/small/test/',
        help='dir of test txt files')

    parser.add_argument(
        "--sub_dataset",
        default="small",
        type=str,
        help="name of sub dataset of LibriLight",
        choices=['small', 'medium', 'large', 'duplicate'], )
    parser.add_argument("--nshard", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    print(f"nshard: {args.nshard}, rank: {args.rank}")

    train_txt_dir = Path(args.train_txt_dir)
    dev_txt_dir = Path(args.dev_txt_dir)
    test_txt_dir = Path(args.test_txt_dir)

    dump_dir = Path(args.dump_dir).expanduser()
    # use absolute path
    dump_dir = dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    train_txt_file = train_txt_dir / f'txt_{args.rank}_{args.nshard}.npy'
    dev_txt_file = dev_txt_dir / f'txt_{args.rank}_{args.nshard}.npy'
    test_txt_file = test_txt_dir / f'txt_{args.rank}_{args.nshard}.npy'

    train_txts = read_txts(train_txt_file)
    dev_txts = read_txts(dev_txt_file)
    test_txts = read_txts(test_txt_file)

    sub_dataset_dump_dir = dump_dir / args.sub_dataset
    sub_dataset_dump_dir.mkdir(parents=True, exist_ok=True)
    train_dump_dir = sub_dataset_dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = sub_dataset_dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = sub_dataset_dump_dir / "test"
    test_dump_dir.mkdir(parents=True, exist_ok=True)
    phonemizer = GruutPhonemizer(language='en-us')

    # process for the 3 sections
    if train_txts:
        process_sentences(
            args=args,
            items=train_txts,
            output_dir=train_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)
    if dev_txts:
        process_sentences(
            args=args,
            items=dev_txts,
            output_dir=dev_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)
    if test_txts:
        process_sentences(
            args=args,
            items=test_txts,
            output_dir=test_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
