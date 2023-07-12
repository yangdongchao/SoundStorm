import os
import numpy as np
data = [['item_name', 'semantic_audio']]

path = "/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_libritts_zql_base360/train/semantic_token"
filenames = []
for filename in os.listdir(path):
    filenames.append(filename)
print(len(filenames))
filenames.sort()
for i, filename in enumerate(filenames):
    if i % 1000 == 0:
        print(i/1000)
    utt_id = filename.split(".")[0]
    semantic_token = np.load(path + "/" +filename)[0].tolist()
    semantic_token_str = ' '.join(str(x) for x in semantic_token)
    data.append([utt_id, semantic_token_str])

   
delimiter = '\t'
filename = "/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_libritts_zql_base360/train/semantic_token.tsv"
with open(filename, 'w', encoding='utf-8') as writer:
    for row in data:
        line = delimiter.join(row)  # 使用制表符拼接每行数据
        writer.write(line + '\n')