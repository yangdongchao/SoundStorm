#!/bin/bash
stage=203
stop_stage=203
root_dir=$1
data_dir=$2
hubert_path=$3
quantizer_path=$4
layer=$5
dump_dir=$6

# please download VAD file for LibriLight first, see ../README.md
# sort spk_id list first by lexicographical order
# split sorted spk_id list into ${nshard}, ${rank} should be in 0 ~ ${nshard} - 1

# get semantic token, download Hubert to pretrained_model/hubert/
# get semantic for small (489 speakers)
# ${nshard} can be 1 for small
# --num-cpu=256 cost 50G GPU of A100
# duplicate 的时候 256 会 OOM
# cost ~10 mins
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=small \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=256 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=0 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=small \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=256 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=1 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=small \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=256 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=2
fi

# get semantic for medium (1596 speakers)
# cost ~ 2.5 hours
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=256 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=0 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=256 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=1 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=256 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=2 & CUDA_VISIBLE_DEVICES=3 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=256 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=3
fi

# get semantic for large (6875 speakers)
if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=0 & CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=1 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=2 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=3 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=4 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=5 & CUDA_VISIBLE_DEVICES=3 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=6 & CUDA_VISIBLE_DEVICES=3 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=7
fi


if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ]; then
CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=8 & CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=9 & CUDA_VISIBLE_DEVICES=5 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=10 & CUDA_VISIBLE_DEVICES=5 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=11 & CUDA_VISIBLE_DEVICES=6 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=12 & CUDA_VISIBLE_DEVICES=6 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=13 & CUDA_VISIBLE_DEVICES=7 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=14 & CUDA_VISIBLE_DEVICES=7 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=112 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=15
fi

# get semantic for duplicate (1100 speakers)
# cost ~ 2.5 hours
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=200 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=0 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=200 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=1 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=200 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=2 & CUDA_VISIBLE_DEVICES=3 python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=200 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=3
fi

# merge semantic tokens
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # input file path list (small、medium、large、duplicate)
    python3 ${BIN_DIR}/merge_semantic_token.py
fi

# extract acoustic token by HiFi-Codec `.pth`
# download hificodec's param to pretrained_model/hificodec/
# softlink AcademiCodec/academicodec to ${MAIN_ROOT} first

# get acoustic for small
# num-cpu=30 for 80G GPU, cost ~ 40 mins
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=small \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=30 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=0 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=small \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=30 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=1 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=small \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=30 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=2
fi

# get acoustic for medium
# cost ~ 5 hours
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=30 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=0 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=30 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=1 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=30 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=2 & CUDA_VISIBLE_DEVICES=3 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=30 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=3
fi

# get acoustic for large
# cost ~ 45 hours
if [ ${stage} -le 200 ] && [ ${stop_stage} -ge 200 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=0 & CUDA_VISIBLE_DEVICES=0 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=1 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=2 & CUDA_VISIBLE_DEVICES=1 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=3 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=4 & CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=5 & CUDA_VISIBLE_DEVICES=3 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=6 & CUDA_VISIBLE_DEVICES=3 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=7
fi


if [ ${stage} -le 201 ] && [ ${stop_stage} -ge 201 ]; then
CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=8 & CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=9 & CUDA_VISIBLE_DEVICES=5 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=10 & CUDA_VISIBLE_DEVICES=5 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=11 & CUDA_VISIBLE_DEVICES=6 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=12 & CUDA_VISIBLE_DEVICES=6 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=13 & CUDA_VISIBLE_DEVICES=7 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=14 & CUDA_VISIBLE_DEVICES=7 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=15
fi

if [ ${stage} -le 202 ] && [ ${stop_stage} -ge 202 ]; then
CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=8 & CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=9 & CUDA_VISIBLE_DEVICES=6 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=12 & CUDA_VISIBLE_DEVICES=6 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=13 
fi

if [ ${stage} -le 203 ] && [ ${stop_stage} -ge 203 ]; then
CUDA_VISIBLE_DEVICES=7 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=12 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=16 \
        --rank=14

fi

# get acoustic for duplicate
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=24 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=0 & CUDA_VISIBLE_DEVICES=5 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=24 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=1 & CUDA_VISIBLE_DEVICES=6 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=24 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=2 & CUDA_VISIBLE_DEVICES=7 python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=24 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=4 \
        --rank=3
fi

# merge acoustic tokens
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    # input file path list (small、medium、large、duplicate)
    python3 ${BIN_DIR}/merge_acoustic_token.py
fi

# test the generated acoustic_token
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    mkdir -p codebook2wav_output
    # HiFi-Codec
    python3 ${BIN_DIR}/codebook2wav.py \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --input_path=${root_dir}/${dump_dir}/small/test/acoustic_token/hificodec/'zola_robinson_aj_64kb#small#2784#short_poetry_collection_073_librivox_64kb_mp3_5.npy' \
        --output_dir=codebook2wav_output/ \
        # --num_quant=3 # NOT WORK HERE, default Nq of HiFi-Codec is 4 and cannot be reduced
fi
