#!/bin/bash

config_path=$1
train_output_path=$2
root_dir=$3
log_frequency=$4
dist_url=$5
dump_dir=$6

omp_num=8

OMP_NUM_THREADS=${omp_num} python3 ${BIN_DIR}/train.py \
        --config_file=${config_path} \
        --train_semantic_path=${root_dir}/${dump_dir}/train/semantic_token.npy \
        --train_acoustic_path=${root_dir}/${dump_dir}/train/acoustic_token/hificodec.pth \
        --dev_semantic_path=${root_dir}/${dump_dir}/dev/semantic_token.npy \
        --dev_acoustic_path=${root_dir}/${dump_dir}/dev/acoustic_token/hificodec.pth \
        --output=${root_dir}/${train_output_path} \
        --log_frequency=${log_frequency} \
        --dist_url=${dist_url} \
        --hificodec_model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --hificodec_config_path=pretrained_model/hificodec/config_16k_320d.json