#!/bin/bash
# test with test set

config_path=$1
train_output_path=$2
ckpt_name=$3
root_dir=$4
dump_dir=$5

python3 ${BIN_DIR}/test.py \
        --config_file=${config_path} \
        --ckpt_path=${root_dir}/${train_output_path}/checkpoint/${ckpt_name} \
        --test_semantic_path=${root_dir}/${dump_dir}/small/test/semantic_token_0_3.npy \
        --test_acoustic_path=${root_dir}/${dump_dir}/small/test/acoustic_token/hificodec_0_3.pth \
        --output_dir=${root_dir}/${train_output_path}/test_output \
        --hificodec_model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --hificodec_config_path=pretrained_model/hificodec/config_16k_320d.json