#!/bin/bash
stage=6
stop_stage=6
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
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=small \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=20 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=0
fi

# get semantic for medium (1596 speakers)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=20 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=0
fi

# get semantic for large (6875 speakers)
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=20 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=12 \
        --rank=0
fi

# get semantic for duplicate (1100 speakers)
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python3 ${BIN_DIR}/get_semantic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=20 \
        --layer=${layer} \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=0
fi

# merge semantic tokens
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # input file path list (small、medium、large、duplicate)
    python3 ${BIN_DIR}/merge_semantic_token.py
fi

# extract acoustic token by HiFi-Codec `.pth`
# download hificodec's param to pretrained_model/hificodec/
# softlink AcademiCodec/academicodec to ${MAIN_ROOT} first

# get acoustic for small
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=small \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=20 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=0
fi

# get acoustic for medium
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=medium \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=20 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=0
fi

# get acoustic for large
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=large \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=20 \
        --spk_num=200 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=12 \
        --rank=0
fi

# get acoustic for duplicate
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    python3 ${BIN_DIR}/get_acoustic_token_librilight.py \
        --data_dir=${data_dir} \
        --sub_dataset=duplicate \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=20 \
        --VAD_path=VAD/librilight_segment_dict.npy \
        --nshard=3 \
        --rank=0
fi

# merge acoustic tokens
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    # input file path list (small、medium、large、duplicate)
    python3 ${BIN_DIR}/merge_acoustic_token.py
fi

# test the generated acoustic_token
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    mkdir -p codebook2wav_output
    # HiFi-Codec
    python3 ${BIN_DIR}/codebook2wav.py \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --input_path=${root_dir}/${dump_dir}/test/acoustic_token/hificodec/986_129388_000067_000000.npy \
        --output_dir=codebook2wav_output/ \
        # --num_quant=3 # NOT WORK HERE, default Nq of HiFi-Codec is 4 and cannot be reduced
fi
