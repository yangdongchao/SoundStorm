#!/bin/bash
stage=0
stop_stage=1
root_dir=$1
data_dir=$2
dump_dir=$3


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # 需要处理不同数据集格式
    python3 ${BIN_DIR}/get_phones.py \
        --data_dir=${data_dir} \
        --dataset=ljspeech \
        --dump_dir=${root_dir}/${dump_dir} \
        --num-cpu=20
fi

# then generate semantic_token (hificodec.pth) in egs1