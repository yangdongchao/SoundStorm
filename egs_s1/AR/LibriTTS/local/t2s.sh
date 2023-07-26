#!/bin/bash
# test with test set

config_path=$1
train_output_path=$2
ckpt_name=$3
root_dir=$4
hubert_path=$5
quantizer_path=$6
prompt_wav_path=$7

python3 ${BIN_DIR}/t2s.py \
    --config_file=${config_path} \
    --ckpt_path=${root_dir}/${train_output_path}/ckpt/${ckpt_name} \
    --output_dir=${root_dir}/${train_output_path}/S1_t2s_output \
    --text_file=${BIN_DIR}/text.txt \
    --hubert_path=${hubert_path} \
    --quantizer_path=${quantizer_path} \
    --prompt_wav_path=${prompt_wav_path}
