#!/bin/bash

config_path=$1
train_output_path=$2
root_dir=$3
log_frequency=$4

python3 ${BIN_DIR}/train.py \
        --config_file=${config_path} \
        --train_semantic_path=${root_dir}/dump/train/semantic_token.tsv \
        --train_acoustic_path=${root_dir}/dump/train/acoustic_token/hificodec.pth \
        --dev_semantic_path=${root_dir}/dump/dev/semantic_token.tsv \
        --dev_acoustic_path=${root_dir}/dump/dev/acoustic_token/hificodec.pth \
        --output=${root_dir}/${train_output_path} \
        --log_frequency=${log_frequency}  