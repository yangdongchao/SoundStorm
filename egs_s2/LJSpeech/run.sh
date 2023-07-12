#!/bin/bash
set -e

source path.sh

gpus=6,7
stage=0
stop_stage=100
train_output_path=exp/default
# dir to set part/all of dump dataset and experiment result
root_dir='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm'
# there should be *.wav 、*/*.wav or */*/*.wav in data_dir
data_dir='~/datasets/LJSpeech-1.1'
config_path=conf/default.yaml
log_frequency=10
# 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
dist_url='tcp://127.0.0.1:29500'
# use which checkpoint file to test
ckpt_name=last.pth
# should be same with ${layer} in hubert_kms.sh
layer=10
hubert_path=pretrained_model/hubert/hubert_base_ls960.pt
quantizer_path=pretrained_model/hubert/hubert_base_ls960_L9_km500.bin


# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${root_dir} ${data_dir} ${hubert_path} ${quantizer_path} ${layer}|| exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${config_path} ${train_output_path} ${root_dir} ${log_frequency} ${dist_url}|| exit -1
fi
# test with test dataset, prompt and target should be the same audio
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/test.sh ${config_path} ${train_output_path} ${ckpt_name} ${root_dir}|| exit -1
fi

# synthesize input prompt/target_semantic/acoustic 4 files, which means prompt and target can from different audios
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh \
    ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} \
    ${hubert_path} ${quantizer_path}|| exit -1
fi

# synthesize_e2e with S1 (text -> semantic token) model
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh || exit -1
fi