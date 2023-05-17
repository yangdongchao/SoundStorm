import os

string = "python SoundStorm/train_spec.py \
         --name speartts_diffsound \
         --config_file SoundStorm/configs/SpearTTS_diffsound.yaml \
         --tensorboard \
         --output SoundStorm/exp_output"

os.system(string)

