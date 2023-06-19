#!/bin/bash
stage=0
stop_stage=0
root_dir=$1
data_dir=$2

# extract semantic token by mHubert `.tsv`
# 需要先下载 mHubert 到指定位置
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # 需要处理不同数据集格式
    python3 ${BIN_DIR}/get_semantic_token.py \
        --data_dir=${data_dir} \
        --dataset=ljspeech \
        --dump_dir=${root_dir}/dump \
        --hubert_path=pretrained_model/mhubert/mhubert_base_vp_en_es_fr_it3.pt \
        --quantizer_path=pretrained_model/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin \
        --num-cpu=10
fi

# extract acoustic token by HiFi-Codec `.pth`
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/get_acoustic_token.py
  
fi

# align the lengths of semantic token and acoustic token
# 如何对齐，剪裁掉长的部分吗？
# 需要看下在训练时的 dataset 是否有对齐的操作
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  
fi

