#!/bin/bash
set -e

source path.sh
# this model is trained with cpu
gpus=4,5,6,7
stage=0
stop_stage=100
# dir to set part/all of dump dataset and experiment result
root_dir='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm'
data_dir='~/datasets/LibriTTS-R/'
sub_dataset_name=train-clean-100
layer=10
n_clusters=1024


# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1
# get tsv file
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/get_tsv_file.py \
        --data_dir=${data_dir} \
        --sub_dataset_name=${sub_dataset_name} \
        --dump_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name} \
        --tsv_name=audio_files.tsv
fi
# dump hubert feature
# generate in parallel
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/dump_hubert_feature.py \
        --tsv_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name} \
        --split=audio_files \
        --ckpt_path=pretrained_model/hubert/hubert_base_ls960.pt \
        --feat_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name}/semantic_feature \
        --layer=${layer} \
        --nshard=5 \
        --rank=0 & CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/dump_hubert_feature.py \
        --tsv_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name} \
        --split=audio_files \
        --ckpt_path=pretrained_model/hubert/hubert_base_ls960.pt \
        --feat_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name}/semantic_feature \
        --layer=${layer} \
        --nshard=5 \
        --rank=1 & CUDA_VISIBLE_DEVICES=5 python3 ${BIN_DIR}/dump_hubert_feature.py \
        --tsv_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name} \
        --split=audio_files \
        --ckpt_path=pretrained_model/hubert/hubert_base_ls960.pt \
        --feat_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name}/semantic_feature \
        --layer=${layer} \
        --nshard=5 \
        --rank=2 & CUDA_VISIBLE_DEVICES=6 python3 ${BIN_DIR}/dump_hubert_feature.py \
        --tsv_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name} \
        --split=audio_files \
        --ckpt_path=pretrained_model/hubert/hubert_base_ls960.pt \
        --feat_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name}/semantic_feature \
        --layer=${layer} \
        --nshard=5 \
        --rank=3 & CUDA_VISIBLE_DEVICES=7 python3 ${BIN_DIR}/dump_hubert_feature.py \
        --tsv_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name} \
        --split=audio_files \
        --ckpt_path=pretrained_model/hubert/hubert_base_ls960.pt \
        --feat_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name}/semantic_feature \
        --layer=${layer} \
        --nshard=5 \
        --rank=4
fi
# learn kmeans
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 ${BIN_DIR}/learn_kmeans.py \
        --feat_dir=${root_dir}/dump_libritts/libritts_${sub_dataset_name}/semantic_feature \
        --nshard=5 \
        --split=audio_files \
        --n_clusters=${n_clusters} \
        --km_path=${root_dir}/dump_libritts/libritts_${sub_dataset_name}/hubert_base_ls960_L${layer}_km${n_clusters}.bin
fi



