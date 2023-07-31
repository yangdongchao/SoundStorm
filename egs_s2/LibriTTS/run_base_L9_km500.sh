#!/bin/bash
set -e

source path.sh

gpus=6,7
stage=0
stop_stage=100
train_output_path='exp_libritts/30k_basex2_base_L9_km500'
# dir to set part/all of dump dataset and experiment result
root_dir='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm'
# there should be *.wav 、*/*.wav or */*/*.wav in data_dir
data_dir='~/datasets/LibriTTS-R'
config_path='conf/30k_basex2_hubert_L9km500.yaml'
log_frequency=1
# 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
dist_url='tcp://127.0.0.1:29501'
# use which checkpoint file to test
ckpt_name='000301e_471119iter.pth'
# should be same with ${layer} in hubert_kms.sh
layer=9
# should be same with ${hubert_path} in hubert_kms.sh
hubert_path=pretrained_model/hubert/hubert_base_ls960.pt
quantizer_path=pretrained_model/hubert/hubert_base_ls960_L9_km500.bin
dump_dir=dump_libritts_base_L9_km500
# for synthesize_e2e.sh
prompt_wav_path='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_libritts_base_L9_km500/test/synthesize_input/LJ050-0179.wav'
S1_config_file='../../egs_s1/AR/LibriTTS/conf/base_L9bin500.yaml'
S1_ckpt_path='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/ar_s1/SoundStorm/exp/base_L9_km500/ckpt/epoch=99-step=49000.ckpt'
sil_token=193 #193 for 500 bin 

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${root_dir} ${data_dir} ${hubert_path} ${quantizer_path} ${layer} ${dump_dir}|| exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${config_path} ${train_output_path} ${root_dir} ${log_frequency} ${dist_url} ${dump_dir}|| exit -1
fi
# test with test dataset, prompt and target should be the same audio
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/test.sh ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} ${dump_dir}|| exit -1
fi

# synthesize input prompt/target_semantic/acoustic 4 files, which means prompt and target can from different audios
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh \
    ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} \
    ${hubert_path} ${quantizer_path} ${dump_dir}|| exit -1
fi

# synthesize_e2e with S1 (text -> semantic token) model
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh \
    ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} \
    ${hubert_path} ${quantizer_path} ${prompt_wav_path} \
    ${S1_config_file} ${S1_ckpt_path} ${sil_token} || exit -1
fi
