import os

import h5py
import pandas as pd
import torch
import torchaudio
from audio_processing.audio_tokenizer import AudioTokenizer
from text_processing.phonemizer import GruutPhonemizer
from tqdm import tqdm

max_duration = 20
min_duration = 3
min_speed = 8
max_speed = 30


def _adjust_begin(audio: torch.Tensor, sr=24000, win_len=200, wait_time=0.05):
    assert (audio.size(0) == 1)
    audio_len = audio.size(1)
    max_audio = audio.abs().max()
    cut = 0
    for i in range(audio_len // win_len - 1):
        wavclip = audio[:, (i * win_len):(i * win_len + win_len)]
        if wavclip.max().item() > 0.05 * max_audio:
            cut = i * win_len
            break

    cut = int(cut - wait_time * sr)
    if cut < 0:
        cut = 0
    return audio[:, cut:]


def _trim_audio(waveform: torch.Tensor,
                sr: int=24000,
                win_len: int=200,
                wait_time=0.05):
    waveform = torch.flip(waveform, dims=[-1])
    waveform = _adjust_begin(
        waveform, sr=sr, win_len=win_len, wait_time=wait_time)
    waveform = torch.flip(waveform, dims=[-1])
    waveform = _adjust_begin(
        waveform, sr=sr, win_len=win_len, wait_time=wait_time)
    return waveform


def preprocess(data_path: str, output_path: str):

    audio_tokenizer = AudioTokenizer()
    phonemizer = GruutPhonemizer(language='en-us')
    # make sure paths are ready
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, 'wavs'))
        os.mkdir(os.path.join(output_path, 'audio_features'))
    h5f = h5py.File(
        os.path.join(output_path, 'audio_features', "audio_features.hdf5"), "w")
    grp = h5f.create_group('feature')

    audio_file_paths = []
    phonemes_transcriptions = []
    audio_frame_keys = []

    # build dataset
    total_duration = 0
    for speaker in tqdm(os.listdir(data_path)):
        for book in os.listdir(os.path.join(data_path, speaker)):
            book_path = os.path.join(data_path, speaker, book)
            df = pd.read_csv(
                f'{book_path}/{speaker}_{book}.trans.tsv', sep='\t')

            for file, text in zip(df.iloc[:, 0], df.iloc[:, 1]):

                file = file + '.wav'
                text = text.replace('\"', '')

                basename = file
                read_path = os.path.join(book_path, file)
                save_path_frame = basename

                # check duration and speed
                try:
                    wav, sr = torchaudio.load(read_path)
                except Exception:
                    #print(f'error reading {read_path}, skip')
                    continue

                wav = audio_tokenizer.convert_audio(wav, sr,
                                                    audio_tokenizer.sample_rate,
                                                    audio_tokenizer.channels)
                #wav = _trim_audio(wav)
                wav_len_in_sec = wav.size(-1) / audio_tokenizer.sample_rate

                # skip if audio too long or too fast / slow

                if wav_len_in_sec > max_duration or wav_len_in_sec < min_duration:
                    #print(f'{basename} of length {wav_len_in_sec} exceeds {max_duration} seconds, skip ...')
                    continue

                phonemes = phonemizer.phonemize(text, espeak=False)
                phoneme_per_sec = len(phonemes) / wav_len_in_sec
                if phoneme_per_sec < min_speed or phoneme_per_sec > max_speed:
                    #print(f'{basename} of speed {phoneme_per_sec} skipped ...')
                    continue

                #torchaudio.save(os.path.join(output_path, save_path_audio), wav, audio_tokenizer.sample_rate)
                audio_file_paths.append(os.path.join(speaker, book, file))

                # save phonemes and text
                total_duration += wav_len_in_sec
                phonemes_transcriptions.append(phonemes)

                # compute codes and save
                wav = wav.unsqueeze(0)
                with torch.no_grad():
                    codes = audio_tokenizer.encode(wav)[0][0][
                        0]  # n_codebooks, T
                codes = codes.cpu().numpy()
                grp.create_dataset(
                    name=save_path_frame, shape=codes.shape, data=codes)
                audio_frame_keys.append(save_path_frame)

    with open(os.path.join(output_path, 'audio_files.tsv'), 'w') as f:
        f.write(data_path + '\n')
        for line in audio_file_paths:
            f.write(line + '\n')

    data_dict = {
        'PathToFile': audio_file_paths,
        'Phonemes': phonemes_transcriptions,
        'AudioFramesPath': audio_frame_keys
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(f'{output_path}/metadata.csv', sep='|', index=False)
    print(f'processed dataset of {total_duration / 3600} hours')
    h5f.flush()
    h5f.close()


if __name__ == '__main__':
    preprocess(
        '/home/yufei/data/LibriTTS/dev-clean',
        'data/libritts_dev_clean', )
