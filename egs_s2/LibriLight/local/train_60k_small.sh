#!/bin/bash

config_path=$1
train_output_path=$2
root_dir=$3
log_frequency=$4
dist_url=$5
dump_dir=$6

omp_num=8

# 注意 *_dirs 参数后面不可以有 ''='
OMP_NUM_THREADS=${omp_num} python3 ${BIN_DIR}/train_librilight_60k.py \
        --config_file=${config_path} \
        --train_semantic_dirs ''${root_dir}'/'${dump_dir}'/small/train/' \
        --train_acoustic_dirs ''${root_dir}'/'${dump_dir}'/small/train/acoustic_token/' \
        --dev_semantic_dirs ''${root_dir}'/'${dump_dir}'/small/dev/' \
        --dev_acoustic_dirs ''${root_dir}'/'${dump_dir}'/small/dev/acoustic_token/' \
        --output=${root_dir}/${train_output_path} \
        --log_frequency=${log_frequency} \
        --dist_url=${dist_url} \
        --hificodec_model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --hificodec_config_path=pretrained_model/hificodec/config_16k_320d.json 