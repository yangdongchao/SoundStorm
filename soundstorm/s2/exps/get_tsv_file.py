# get tsv file for dump_hubert_feature.py
import argparse
import os
from pathlib import Path


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="get tsv file for dump_hubert_fature")

    parser.add_argument(
        "--data_dir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--sub_dataset_name", default="train-clean-360", type=str)

    parser.add_argument(
        "--dump_dir", type=str, help="directory to dump feature files.")

    parser.add_argument("--tsv_name", type=str)

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    # use absolute path
    dump_dir = dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    assert data_dir.is_dir()
    print("data_dir:", data_dir)

    sub_dataset_dir = data_dir / args.sub_dataset_name
    prefix = str(sub_dataset_dir) + '/'
    print("sub_dataset_dir:", sub_dataset_dir)
    # filter out hidden files
    speaker_list = [
        file for file in os.listdir(sub_dataset_dir) if not file.startswith('.')
    ]
    all_wav_files = []
    for speaker in speaker_list:
        wav_files = sorted(list((sub_dataset_dir / speaker).rglob("*/*.wav")))
        # filter out ._*.wav
        wav_files = [
            file for file in wav_files if not file.name.startswith('._')
        ]
        all_wav_files.extend(wav_files)

    all_wav_files = list(map(str, all_wav_files))
    all_wav_files = [remove_prefix(text, prefix) for text in all_wav_files]

    print("len(all_wav_files):", len(all_wav_files))
    with open(dump_dir / args.tsv_name, 'w', encoding='utf-8') as writer:
        writer.write(prefix + '\n')
        for wav_file in all_wav_files:
            writer.write(wav_file + '\n')


if __name__ == "__main__":
    main()
