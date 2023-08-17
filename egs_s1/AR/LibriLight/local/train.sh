#!/bin/bash

config_path=$1
train_output_path=$2
root_dir=$3
dump_dir=$4


# 注意 *_dirs 参数后面不可以有 ''='
python3 ${BIN_DIR}/train_librilight_6k.py \
    --config_file=${config_path} \
    --train_semantic_dirs ''${root_dir}'/'${dump_dir}'/small/test/' ''${root_dir}'/'${dump_dir}'/medium/test/' \
    --train_phoneme_dirs ''${root_dir}'/'${dump_dir}'/small/test/' ''${root_dir}'/'${dump_dir}'/medium/test/' \
    --dev_semantic_dirs ''${root_dir}'/'${dump_dir}'/small/dev/' ''${root_dir}'/'${dump_dir}'/medium/dev/' \
    --dev_phoneme_dirs ''${root_dir}'/'${dump_dir}'/small/dev/' ''${root_dir}'/'${dump_dir}'/medium/dev/' \
    --output_dir=${root_dir}/${train_output_path}