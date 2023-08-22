from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

VAD_dict_path = '/home/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/egs_s2/LibriLight/VAD/librilight_segment_dict.npy'
data_dir = '/home/yuantian04/datasets/LibriLight'
test_tsv_path = '/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_librilight/small/test/semantic_token_0_3.tsv'
output_dir = '/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_librilight/small/test/synthesize_input'
output_dir = Path(output_dir)
# key_name = f'{wav_name}#{sub_dataset}#{spk_id}#{book_name}'
semantic_data = pd.read_csv(test_tsv_path, delimiter='\t')
print("len(semantic_data):", len(semantic_data))
sr = 16000

VAD_dict = np.load(VAD_dict_path, allow_pickle=True).item()
for i in range(len(semantic_data)):
    # 先依次遍历
    # get str
    if i % 100 == 0:
        print(f'{i}...')
    try:
        key_name = semantic_data['item_name'][i]
        # 移除最后下划线后面的 split 数字
        raw_wav_name = '_'.join(key_name.split('_')[:-1])
        assert raw_wav_name in VAD_dict.keys()
        wav_name, sub_dataset, spk_id, book_name = raw_wav_name.split("#")
        wav_path = f'{data_dir}/{sub_dataset}/{spk_id}/{book_name}/{wav_name}.flac'
        raw_wav, _ = librosa.load(str(wav_path), sr=sr)
        start, end = VAD_dict[raw_wav_name][key_name]
        sub_wav = raw_wav[int(start * sr):int(end * sr)]
        sf.write(output_dir / (key_name + ".wav"), sub_wav, sr)
    except Exception:
        continue
