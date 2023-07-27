import argparse
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from academicodec.models.hificodec.vqvae import VQVAE
from soundstorm.s2.models.dalle_wav.build import build_model
from soundstorm.s2.models.hubert.semantic_tokenizer import SemanticTokenizer
from soundstorm.utils.io import load_yaml_config

acoustic_token_nums = 1024


def move_tensors_to_cuda(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = move_tensors_to_cuda(value)
        elif isinstance(value, torch.Tensor):
            d[key] = value.cuda()
    return d


def hificodec_decode(hificodec, acoustic_token, rescale=True):
    """
    acoustic_token: shape [B, Nq, T]
    """
    # hificodec decode 需要 [B, T, Nq]
    # [B, Nq, T] -> [B, T, Nq]
    acoustic_token = torch.clamp(acoustic_token, 0, acoustic_token_nums - 1)
    acoustic_token = acoustic_token.transpose(1, 2)
    # VQVAE.forward()
    wav = hificodec(acoustic_token)
    # (1, 1, T) -> (T,)
    wav = wav.detach().squeeze().cpu().numpy()
    limit = 0.99
    if rescale:
        mx = np.abs(wav).max()
        if mx != 0:
            wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clip(-limit, limit)
    return wav


# one wav per batch
def get_batch(prompt_semantic_tokens,
              prompt_acoustic_tokens,
              target_semantic_tokens,
              num_quant=4):
    hz = 50
    # transformer_utils.py 里面最大是 20, pad 了一个  stop token, 所以这里最大是 19
    # 但是训练时最多是 10s, 所以超过 10s 的无法合成出来
    max_sec = 10

    # prompt 最多为 3s
    if prompt_acoustic_tokens.shape[1] > 6 * hz:
        prompt_len = 3 * hz
    else:
        prompt_len = prompt_acoustic_tokens.shape[1] // 2

    prompt_semantic_tokens = prompt_semantic_tokens[:, :prompt_len]
    prompt_acoustic_tokens = prompt_acoustic_tokens[:, :prompt_len]
    # target 最多为 10s
    target_semantic_tokens = target_semantic_tokens[:, :max_sec * hz]
    # acoustic_token 和 semantic_token 长度是对齐的
    target_T = target_semantic_tokens.shape[-1]
    # 伪造的 target_acoustics_tokens
    target_acoustics_tokens = torch.zeros([num_quant, target_T])

    # 用 False 指示有值的位置, shape (1, T)
    x_mask = torch.zeros((1, target_T)).bool()

    samples = {}
    # pseudo batch
    samples['prompt_semantics'] = prompt_semantic_tokens.unsqueeze(0)
    samples['target_semantics'] = target_semantic_tokens.unsqueeze(0)
    samples['prompt_acoustics'] = prompt_acoustic_tokens.unsqueeze(0)
    samples['target_acoustics'] = target_acoustics_tokens.unsqueeze(0)
    samples['x_mask'] = x_mask

    return samples


def evaluate(args, hificodec, soundstorm, semantic_tokenizer=None):
    num_quant = 4
    sample_rate = 16000
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_semantic_path is None and args.prompt_acoustic_path is None:
        # get prompt_semantic and prompt_acoustic from prompt_wav
        assert args.prompt_wav_path is not None and semantic_tokenizer is not None
        prompt_name = os.path.basename(args.prompt_wav_path).split('.')[0]
        wav, _ = librosa.load(args.prompt_wav_path, sr=16000)
        wav = torch.tensor(wav).unsqueeze(0)
        wav = wav.cuda()
        # get prompt_semantic
        # (1, T)
        prompt_semantic_tokens = semantic_tokenizer.tokenize(wav)
        print("prompt_semantic_tokens.shape:", prompt_semantic_tokens.shape)

        # get prompt_acoustic
        # (1, T, 4)
        acoustic_token = hificodec.encode(wav)
        # trans acoustic_token.shape to (Nq, T)
        prompt_acoustic_tokens = acoustic_token.squeeze(0).transpose(0, 1)
    else:
        prompt_semantic_data = pd.read_csv(
            args.prompt_semantic_path, delimiter='\t')
        prompt_acoustic_data = torch.load(args.prompt_acoustic_path)

        prompt_name = prompt_semantic_data['item_name'][0]
        prompt_semantic_str = prompt_semantic_data['semantic_audio'][0]
        # shape: (1, T)
        prompt_semantic_tokens = torch.tensor(
            [int(idx) for idx in prompt_semantic_str.split(' ')]).unsqueeze(0)
        try:
            prompt_acoustic_str = prompt_acoustic_data[prompt_name]
        except Exception:
            return None
        prompt_acoustic_tokens = prompt_acoustic_str[:num_quant, ...]

    # get target
    # 保留 item_name 前导 0
    target_semantic_data = pd.read_csv(
        args.target_semantic_path, delimiter='\t', dtype=str)
    target_name = target_semantic_data['item_name'][0]
    target_semantic_str = target_semantic_data['semantic_audio'][0]
    # shape: (1, T)
    target_semantic_tokens = torch.tensor(
        [int(idx) for idx in target_semantic_str.split(' ')]).unsqueeze(0)

    batch = get_batch(
        prompt_semantic_tokens=prompt_semantic_tokens,
        prompt_acoustic_tokens=prompt_acoustic_tokens,
        target_semantic_tokens=target_semantic_tokens,
        num_quant=num_quant)

    batch = move_tensors_to_cuda(batch)

    with torch.no_grad():
        start = time.time()
        model_out = soundstorm.infer_one(batch)
        end = time.time()
        print("infer time:", end - start)
    content = model_out['token_pred']
    # shape (B, Nq x T) -> (B, Nq, T)
    codes = content.reshape(content.shape[0], num_quant, -1)
    wav_gen = hificodec_decode(hificodec, codes)

    sf.write(output_dir / ("s_" + prompt_name + "_t_" + target_name + ".wav"),
             wav_gen, sample_rate)


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(description="Run SoundStorm for test set.")

    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/default.yaml',
        help='path of config file')

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='exp/default/checkpoint/last.pth',
        help='Checkpoint file of SoundStorm model.')

    # args for dataset
    parser.add_argument(
        '--prompt_semantic_path',
        type=str,
        default=None,
        help='should be match with prompt_acoustic')
    parser.add_argument(
        '--prompt_acoustic_path',
        type=str,
        default=None,
        help='should be match with target_acoustic')
    parser.add_argument(
        '--prompt_wav_path',
        type=str,
        default=None,
        help='if prompt_semantic_path and prompt_acoustic_path is None, \
        we should extract them from prompt_wav')
    # to get semantic tokens from prompt_wav
    parser.add_argument("--hubert_path", type=str, default=None)
    parser.add_argument("--quantizer_path", type=str, default=None)
    # the speaker should be different with prompt
    # source 1001_134708_000013_000000 -> 只有一条，所以不在训练集
    # target 98_199_000030_000000
    parser.add_argument(
        '--target_semantic_path',
        type=str,
        default='dump/test/target_semantic.tsv')

    # for HiFi-Codec
    parser.add_argument(
        "--hificodec_model_path",
        type=str,
        default='pretrained_model/hificodec//HiFi-Codec-16k-320d')
    parser.add_argument(
        "--hificodec_config_path",
        type=str,
        default='pretrained_model/hificodec/config_16k_320d.json')

    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # get models
    # get codec
    hificodec = VQVAE(
        config_path=args.hificodec_config_path,
        ckpt_path=args.hificodec_model_path,
        with_encoder=True)
    hificodec.generator.remove_weight_norm()
    hificodec.encoder.remove_weight_norm()
    hificodec.eval()
    hificodec.cuda()

    # get soundstorm
    config = load_yaml_config(args.config_file)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    soundstorm = build_model(config)
    soundstorm.load_state_dict(ckpt["model"])
    soundstorm.eval()
    soundstorm.cuda()

    semantic_tokenizer = None

    if args.hubert_path is not None and args.quantizer_path is not None:
        # get prompt_semantic from prompt_wav
        semantic_tokenizer = SemanticTokenizer(
            hubert_path=args.hubert_path,
            quantizer_path=args.quantizer_path,
            duplicate=True)

    # cost 14s for a 10s target
    evaluate(args, hificodec, soundstorm, semantic_tokenizer)


if __name__ == "__main__":
    main()
