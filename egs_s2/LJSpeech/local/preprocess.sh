#!/bin/bash
stage=1
stop_stage=1
root_dir=$1
data_dir=$2

# extract semantic token by mHubert `.tsv`
# download mHubert to pretrained_model/mhubert/
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # 需要处理不同数据集格式
    python3 ${BIN_DIR}/get_semantic_token.py \
        --data_dir=${data_dir} \
        --dataset=ljspeech \
        --dump_dir=${root_dir}/dump \
        --hubert_path=pretrained_model/mhubert/mhubert_base_vp_en_es_fr_it3.pt \
        --quantizer_path=pretrained_model/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin \
        --num-cpu=20
fi

# extract acoustic token by HiFi-Codec `.pth`
# download hificodec's param to pretrained_model/hificodec/
# softlink AcademiCodec/academicodec to ${MAIN_ROOT} first

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # HiFi-Codec
    python3 ${BIN_DIR}/get_acoustic_token.py \
        --data_dir=${data_dir} \
        --dataset=ljspeech \
        --dump_dir=${root_dir}/dump \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --num-cpu=20

    # Encodec
    # when target_bw=12 for 16k_320d, Nq=24
    python3 ${BIN_DIR}/get_acoustic_token.py \
        --data_dir=${data_dir} \
        --dataset=ljspeech \
        --dump_dir=${root_dir}/dump \
        --codec_name=encodec \
        --model_path=pretrained_model/encodec/encodec_16k_320d.pth \
        --ratios 8 5 4 2 \
        --target_bandwidths 1 1.5 2 4 6 12 \
        --target_bw=12 \
        --sr=16000 \
        --num-cpu=20
  
fi

# align the lengths of semantic token and acoustic token
# 如何对齐，剪裁掉长的部分吗？
# 需要看下在训练时的 dataset 是否有对齐的操作
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "11111"
  
fi

