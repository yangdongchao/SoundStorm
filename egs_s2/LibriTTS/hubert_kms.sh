#!/bin/bash
set -e

source path.sh

gpus=6,7
stage=0
stop_stage=100
train_output_path=hubert_kms
# dir to set part/all of dump dataset and experiment result
root_dir='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm'
# there should be *.wav 、*/*.wav or */*/*.wav in data_dir
data_dir='~/datasets/LibriTTS-R/'
sub_dataset_name=train-clean-100


# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1
# get tsv file
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/get_tsv_file.py \
        --data_dir=${data_dir} \
        --sub_dataset_name=${sub_dataset_name} \
        --dump_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name} \
        --tsv_name=audio_files.tsv
fi
# dump hubert feature
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/dump_hubert_feature.py \
        --tsv_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name} \
        --split=audio_files \
        --ckpt_path=pretrained_model/hubert/hubert_base_ls960.pt \
        --feat_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name}/semantic_feature \
        --layer=10 \
        --nshard=1 \
        --rank=0
fi
# learn kmeans
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
fi
