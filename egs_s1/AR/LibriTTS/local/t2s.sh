#!/bin/bash
# test with test set

config_path=$1
train_output_path=$2
ckpt_name=$3
root_dir=$4

python3 ${BIN_DIR}/t2s.py \
    --config_file=${config_path} \
    --ckpt_path=${root_dir}/${train_output_path}/ckpt/${ckpt_name} \
    --output_dir=${root_dir}/${train_output_path}/S1_t2s_output \
    --text_file=${BIN_DIR}/text.txt