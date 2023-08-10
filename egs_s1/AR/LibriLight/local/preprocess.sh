#!/bin/bash
stage=0
stop_stage=0
root_dir=$1
data_dir=$2
dump_dir=$3

# get text use ASR (whisper small) for LibriLight

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    sub_dataset=small
    echo "get_txt_librilight.py for ${sub_dataset} start!"
    for rank_id in {0..2}; do
        gpu_id=$((rank_id+4))
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 ${BIN_DIR}/get_txt_librilight.py \
            --data_dir=${data_dir} \
            --sub_dataset=${sub_dataset} \
            --dump_dir=${root_dir}/${dump_dir} \
            --num-cpu=340 \
            --VAD_path=VAD/librilight_segment_dict.npy \
            --nshard=3 \
            --rank=${rank_id} &
        eval pid${rank_id}="$!"
    done
    wait "$pid0" "$pid1" "$pid2"
    echo "get_txt_librilight.py for ${sub_dataset} done!"
fi

# get semantic for medium (1596 speakers)
# cost ~ 2.5 hours
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    sub_dataset=medium
    echo "get_txt_librilight.py for ${sub_dataset} start!"
    for rank_id in {0..3}; do
        gpu_id=$((rank_id))
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 ${BIN_DIR}/get_txt_librilight.py \
            --data_dir=${data_dir} \
            --sub_dataset=${sub_dataset} \
            --dump_dir=${root_dir}/${dump_dir} \
            --num-cpu=340 \
            --VAD_path=VAD/librilight_segment_dict.npy \
            --nshard=4 \
            --rank=${rank_id} &
        eval pid${rank_id}="$!"
    done
    wait "$pid0" "$pid1" "$pid2" "$pid3"
    echo "get_txt_librilight.py for ${sub_dataset} done!"
fi

# get semantic for large (6875 speakers)
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    sub_dataset=large
    echo "get_txt_librilight.py for ${sub_dataset} start!"
    for rank_id in {0..15}; do
        gpu_id=$((rank_id / 2))
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 ${BIN_DIR}/get_txt_librilight.py \
            --data_dir=${data_dir} \
            --sub_dataset=${sub_dataset} \
            --dump_dir=${root_dir}/${dump_dir} \
            --num-cpu=170 \
            --VAD_path=VAD/librilight_segment_dict.npy \
            --nshard=16 \
            --rank=${rank_id} &
        eval pid${rank_id}="$!"
    done
    wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5" "$pid6" "$pid7" \
    "$pid8" "$pid9" "$pid10" "$pid11" "$pid12" "$pid13" "$pid14" "$pid15"
    echo "get_txt_librilight.py for ${sub_dataset} done!"
fi

# get semantic for duplicate (1100 speakers)
# cost ~ 2.5 hours
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    sub_dataset=duplicate
    echo "get_txt_librilight.py for ${sub_dataset} start!"
    for rank_id in {0..3}; do
        gpu_id=$((rank_id))
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 ${BIN_DIR}/get_txt_librilight.py \
            --data_dir=${data_dir} \
            --sub_dataset=${sub_dataset} \
            --dump_dir=${root_dir}/${dump_dir} \
            --num-cpu=340 \
            --VAD_path=VAD/librilight_segment_dict.npy \
            --nshard=4 \
            --rank=${rank_id} &
        eval pid${rank_id}="$!"
    done
    wait "$pid0" "$pid1" "$pid2" "$pid3"
    echo "get_txt_librilight.py for ${sub_dataset} done!"
fi

# TODO
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    python3 ${BIN_DIR}/get_phones_librilight.py 
fi

# generate semantic_token (hificodec.pth) in egs1
