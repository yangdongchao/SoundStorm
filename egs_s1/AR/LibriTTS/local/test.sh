#!/bin/bash
# test with test set

config_path=$1
train_output_path=$2
ckpt_name=$3
root_dir=$4
dump_dir=$5

python3 ${BIN_DIR}/test.py \
    --config_file=${config_path} \
    --ckpt_path=${root_dir}/${train_output_path}/ckpt/${ckpt_name} \
    --test_semantic_path=${root_dir}/${dump_dir}/test/semantic_token.tsv \
    --test_phoneme_path=${root_dir}/${dump_dir}/test/phonemes.npy \
    --output_dir=${root_dir}/${train_output_path}/S1_test_output