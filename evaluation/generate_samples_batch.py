# ------------------------------------------
# Diffsound
# written by Dongchao Yang
# code based https://github.com/cientgu/VQ-Diffusion
# ------------------------------------------
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import numpy as np
from sound_synthesis.utils.io import load_yaml_config
from sound_synthesis.modeling.build import build_model
from sound_synthesis.utils.misc import get_model_parameters_info
import datetime
from pathlib import Path
from vocoder.modules import Generator
import yaml
import pandas as pd
import typing as tp
import torchaudio


def save_audio(wav: torch.Tensor,
               path: tp.Union[Path, str],
               sample_rate: int,
               rescale: bool=False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(
        path,
        wav,
        sample_rate=sample_rate,
        encoding='PCM_S',
        bits_per_sample=16)


def load_vocoder(ckpt_vocoder: str, eval_mode: bool):
    ckpt_vocoder = Path(ckpt_vocoder)
    # print('ckpt_vocoder ',ckpt_vocoder)
    vocoder_sd = torch.load(ckpt_vocoder / 'best_netG.pt', map_location='cpu')
    # print('vocoder_sd ',vocoder_sd)
    with open(ckpt_vocoder / 'args.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.UnsafeLoader)
    vocoder = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers)
    vocoder.load_state_dict(vocoder_sd)
    if eval_mode:
        vocoder.eval()
    return {'model': vocoder}


def get_mask_from_lengths(lengths, max_len=None):
    """Get pad mask"""
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size,
                                                       -1).type_as(lengths)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


from omegaconf import OmegaConf


def check_clipping2(wav, rescale):
    if rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model


def build_soundstream():
    config_path = 'encodec_16k_6kbps_v4/config.yaml'
    resume_path = 'encodec_16k_6kbps_v4/model_ckpts/ckpt_01215000.pth'
    config = OmegaConf.load(config_path)
    soundstream = build_codec_model(config)
    # soundstream = SoundStream(n_filters=32, D=512, ratios=[8, 5, 4, 2]) 
    parameter_dict = torch.load(resume_path)
    soundstream.load_state_dict(parameter_dict['codec_model'])  # load model
    soundstream = soundstream.cuda()
    return soundstream


class Diffsound():
    def __init__(self, config, path):
        self.info = self.get_model(
            ema=True, model_path=path, config_path=config)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_model(self, ema, model_path, config_path):
        if 'OUTPUT' in model_path:  # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else:
            model_name = os.path.basename(config_path).replace('.yaml', '')
        config = load_yaml_config(config_path)
        model = build_model(config)  #加载 dalle model
        model_parameters = get_model_parameters_info(model)  #参数详情
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)
        if ema is True and 'ema' in ckpt:
            print("Evaluate EMA model")
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(
                ckpt['ema'], strict=False)
        return {
            'model': model,
            'epoch': epoch,
            'model_name': model_name,
            'parameter': model_parameters
        }

    def read_tsv(self, val_path):
        train_tsv = pd.read_csv(val_path, sep=',', usecols=[0, 1])
        filenames = train_tsv['file_name']
        captions = train_tsv['caption']
        filenames_ls = []
        captions_ls = []
        for name in filenames:
            filenames_ls.append(name)
        for cap in captions:
            captions_ls.append(cap)
        caps_dict = {}
        for i in range(len(filenames_ls)):
            if filenames_ls[i] not in caps_dict.keys():
                caps_dict[filenames_ls[i]] = [captions_ls[i]]
            else:
                caps_dict[filenames_ls[i]].append(captions_ls[i])
        return caps_dict

    def generate_sample(self, val_path, truncation_rate, save_root, fast=False):
        semantic_path = 'semantic/test.tsv'
        acoustic_path = os.path.join('LibriTTS_1000', 'acoustic',
                                     'acoustic_2.pth')
        acoustic_data = torch.load(acoustic_path)  # get dict
        semantic_data = pd.read_csv(
            semantic_path, delimiter='\t')  # 读取 semantic
        import time
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        store_root = os.path.join(save_root, 'generate_' + time_str)
        os.makedirs(store_root, exist_ok=True)
        soundstream = build_soundstream()
        for index in range(len(semantic_data['item_name'])):
            name = semantic_data['item_name'][index]
            semantic_str = semantic_data['tgt_audio'][index]
            semantic_tokens = torch.tensor(
                [int(idx) for idx in semantic_str.split(' ')]).unsqueeze(0)
            print('semantic_tokens ', semantic_tokens.shape)
            acoustic_tokens = acoustic_data[name]  # 
            acoustic_tokens = torch.from_numpy(acoustic_tokens)
            acoustic_tokens = acoustic_tokens.squeeze(1)  # n,len
            acoustic_tokens = acoustic_tokens.unsqueeze(0)  # 
            acoustic_tokens = acoustic_tokens[:, :
                                              3, :]  # only use 3 codebook, you can set any config.
            print('acoustic_tokens ', acoustic_tokens.shape)
            if acoustic_tokens.shape[1] > 6 * 50:
                tmp_len = 150
            else:
                tmp_len = acoustic_tokens.shape[1] // 2  #
            prompt_acoustic_tokens = acoustic_tokens[:, :, :tmp_len]
            prompt_semantic_tokens = semantic_tokens[:, :tmp_len]
            target_semantic_tokens = semantic_tokens[:, tmp_len:tmp_len +
                                                     500]  # the left 

            prompt_semantic_tokens = prompt_semantic_tokens.cuda()
            target_semantic_tokens = target_semantic_tokens.cuda()
            prompt_acoustic_tokens = prompt_acoustic_tokens.cuda()
            real_acoustic_tokens = acoustic_tokens[:, :, tmp_len:tmp_len +
                                                   500]  # 
            real_acoustic_tokens = real_acoustic_tokens.cuda()
            if fast is not False:
                add_string = 'r,fast' + str(fast - 1)
            else:
                add_string = 'r'
            data_i = {}
            x_mask = torch.ones((1, real_acoustic_tokens.shape[-1]))
            x_mask = (x_mask == 0)
            # print('x_mask ', x_mask)
            new_samples = {}
            new_samples['prompt_semantics'] = prompt_semantic_tokens
            new_samples['target_semantics'] = target_semantic_tokens
            new_samples['prompt_acoustics'] = prompt_acoustic_tokens
            new_samples['target_acoustics'] = real_acoustic_tokens
            new_samples['x_mask'] = x_mask.cuda()
            with torch.no_grad():
                model_out = self.model.generate_content_tmp(
                    batch=new_samples,
                    filter_ratio=0,
                    replicate=1,  # 每个样本重复多少次?
                    content_ratio=1,
                    return_att_weight=False,
                    sample_type="top" + str(truncation_rate) + add_string,
                )  # B x C x H x W
                content = model_out['token_pred']  #
                codes = content.reshape(content.shape[0], 3,
                                        -1)  # reshape to original shape
                codes = codes.transpose(0, 1)
                # print('content ', content.shape)
                # assert 1==2
                #ge_acoustic_tokens = codes.permute(2,0,1)
                out = soundstream.decode(codes)
                print('out ', out.shape)
                coarse_wav = out.detach().cpu().squeeze(0)
                print('real_acoustic_tokens ', real_acoustic_tokens.shape)
                real_out = soundstream.decode(
                    real_acoustic_tokens.transpose(0, 1))
                real_wav = real_out.detach().cpu().squeeze(0)
                save_audio(coarse_wav.cpu(), store_root + '/' + name + '.wav',
                           16000)
                save_audio(real_wav.cpu(),
                           store_root + '/' + name + '_real.wav', 16000)
                #assert 1==2


if __name__ == '__main__':
    # Note that cap_text.yaml includes the config of vagan, we must choose the right path for it.
    config_path = './diff_instruct.yaml'
    pretrained_model_path = './000399e_404399iter.pth'
    save_root_ = './generated_sample'
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    key_words = 'instructtts_diffsound'
    now = (datetime.datetime.now() - random_seconds_shift
           ).strftime('%Y-%m-%dT%H-%M-%S')
    save_root = os.path.join(save_root_, key_words + '_samples_' + now)
    os.makedirs(save_root, exist_ok=True)

    audio_path = ''  # the audio path
    Diffsound = Diffsound(config=config_path, path=pretrained_model_path)
    Diffsound.generate_sample(
        audio_path, truncation_rate=0.85, save_root=save_root, fast=False)
