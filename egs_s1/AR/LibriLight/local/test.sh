#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3
root_dir=$4
dump_dir=$5
omp_num=4

OMP_NUM_THREADS=${omp_num} python3 ${BIN_DIR}/test.py \
    --config_file=${config_path} \
    --ckpt_path=${root_dir}/${train_output_path}/ckpt/${ckpt_name} \
    --test_semantic_path=${root_dir}/${dump_dir}/small/test/semantic_token_0_3.npy \
    --test_phoneme_path=${root_dir}/${dump_dir}/small/test/phonemes_0_3.npy \
    --output_dir=${root_dir}/${train_output_path}/S1_test_output