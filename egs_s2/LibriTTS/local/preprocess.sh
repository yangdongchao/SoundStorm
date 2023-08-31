#!/bin/bash
stage=0
stop_stage=0
root_dir=$1
data_dir=$2
hubert_path=$3
quantizer_path=$4
layer=$5
dump_dir=$6

# extract semantic token by mHubert `.tsv`
# download Hubert to pretrained_model/hubert/
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # 需要处理不同数据集格式
    OMP_NUM_THREADS=1 python3 ${BIN_DIR}/get_semantic_token.py \
        --data_dir=${data_dir} \
        --dataset=libritts \
        --dump_dir=${root_dir}/${dump_dir} \
        --hubert_path=${hubert_path} \
        --quantizer_path=${quantizer_path} \
        --num-cpu=256 \
        --layer=${layer}
fi

# extract acoustic token by HiFi-Codec `.pth`
# download hificodec's param to pretrained_model/hificodec/
# softlink AcademiCodec/academicodec to ${MAIN_ROOT} first

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # HiFi-Codec
    OMP_NUM_THREADS=1 python3 ${BIN_DIR}/get_acoustic_token.py \
        --data_dir=${data_dir} \
        --dataset=libritts \
        --dump_dir=${root_dir}/${dump_dir} \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=20

    # # Encodec
    # # when target_bw=12 for 16k_320d, Nq=24
    # python3 ${BIN_DIR}/get_acoustic_token.py \
    #     --data_dir=${data_dir} \
    #     --dataset=libritts \
    #     --dump_dir=${root_dir}/${dump_dir} \
    #     --codec_name=encodec \
    #     --model_path=pretrained_model/encodec/encodec_16k_320d.pth \
    #     --ratios 8 5 4 2 \
    #     --target_bandwidths 1 1.5 2 4 6 12 \
    #     --target_bw=12 \
    #     --sr=16000 \
    #     --num-cpu=20
fi

# test the generated acoustic_token
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
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
    
    # # Encodec
    # # when target_bw=12 for 16k_320d, Nq=24
    # python3 ${BIN_DIR}/codebook2wav.py \
    #     --codec_name=encodec \
    #     --model_path=pretrained_model/encodec/encodec_16k_320d.pth \
    #     --ratios 8 5 4 2 \
    #     --target_bandwidths 1 1.5 2 4 6 12 \
    #     --target_bw=12 \
    #     --sr=16000 \
    #     --input_path=${root_dir}/${dump_dir}/test/acoustic_token/hificodec/986_129388_000067_000000.npy  \
    #     --output_dir=codebook2wav_output/ \
    #     # --num_quant=3 # default Nq of Encodec is 24
fi
