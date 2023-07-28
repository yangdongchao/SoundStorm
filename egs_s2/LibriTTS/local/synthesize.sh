#!/bin/bash
# synthesize input 需要自己构造一下三个输入用于测试，最好把原始 wav 找到
# 1. prompt_semantic
# 2. prompt_acoustic
# 3. target_semantic
# 1 和 2 需要是同一个音频提取的

config_path=$1
train_output_path=$2
ckpt_name=$3
root_dir=$4

hubert_path=$5
quantizer_path=$6

dump_dir=$7

stage=0
stop_stage=0

# input prompt_semantic and prompt_acoustic 
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/synthesize.py \
        --config_file=${config_path} \
        --ckpt_path=${root_dir}/${train_output_path}/checkpoint/${ckpt_name} \
        --prompt_semantic_path=${root_dir}/${dump_dir}/test/synthesize_input/prompt_semantic.tsv \
        --prompt_acoustic_path=${root_dir}/${dump_dir}/test/synthesize_input/prompt_acoustic.pth \
        --target_semantic_path=${root_dir}/${dump_dir}/test/synthesize_input/target_semantic.tsv \
        --output_dir=${root_dir}/${train_output_path}/syn_output \
        --hificodec_model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d \
        --hificodec_config_path=pretrained_model/hificodec/config_16k_320d.json
fi

# input prompt_semantic and prompt_acoustic 
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/synthesize.py \
        --config_file=${config_path} \
        --ckpt_path=${root_dir}/${train_output_path}/checkpoint/${ckpt_name} \
        --prompt_wav_path=${root_dir}/${dump_dir}/test/synthesize_input/1001_134708_000013_000000.wav \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --target_semantic_path=${root_dir}/${dump_dir}/test/synthesize_input/target_semantic.tsv \
        --output_dir=${root_dir}/${train_output_path}/syn_output \
        --hificodec_model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d \
        --hificodec_config_path=pretrained_model/hificodec/config_16k_320d.json
fi