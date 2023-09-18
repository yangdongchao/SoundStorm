#!/bin/bash
# for S2
config_path=$1
train_output_path=$2
ckpt_name=$3
root_dir=$4
# for codec and prompt
hubert_path=$5
quantizer_path=$6
prompt_wav_path=$7
# for S1
S1_config_file=$8
S1_ckpt_path=$9
sil_token=${10}

omp_num=4

OMP_NUM_THREADS=${omp_num} python3 ${BIN_DIR}/synthesize_e2e.py \
    --S2_config_file=${config_path} \
    --S2_ckpt_path=${root_dir}/${train_output_path}/checkpoint/${ckpt_name} \
    --S1_config_file=${S1_config_file} \
    --S1_ckpt_path=${S1_ckpt_path} \
    --prompt_wav_path=${prompt_wav_path} \
    --text_file=${BIN_DIR}/text.txt \
    --hubert_path=${hubert_path} \
    --quantizer_path=${quantizer_path} \
    --output_dir=${root_dir}/${train_output_path}/syn_e2e_output \
    --hificodec_model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
    --hificodec_config_path=pretrained_model/hificodec/config_16k_320d.json \
    --sil_token=${sil_token}
