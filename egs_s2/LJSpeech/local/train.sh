#!/bin/bash

python3 ${BIN_DIR}/train.py \
         --name speartts_diffsound \
         --config_file SoundStorm/configs/SpearTTS_diffsound.yaml \
         --tensorboard \
         --output SoundStorm/exp_output