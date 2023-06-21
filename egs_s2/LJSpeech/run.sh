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
data_dir='~/datasets/LJSpeech_mini'
config_path=conf/default.yaml


# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ./local/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${root_dir} ${data_dir}|| exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${config_path} ${train_output_path} ${root_dir}|| exit -1
fi
# synthesize with test dataset
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/test.sh ${train_output_path} ${ptfile_name} ${root_dir} ${config_path} ${model}|| exit -1
fi
# synthesize_e2e
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${train_output_path} ${ptfile_name} ${root_dir} ${config_path} ${model}|| exit -1
fi