import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from academicodec.models.hificodec.vqvae import VQVAE
from soundstorm.s2.models.dalle_wav.build import build_model
from soundstorm.s2.utils.io import load_yaml_config

# 每一条构成一个 batch 过一遍模型
# 是单条推理还是凑 batch 推理？=> 可以实现两种分别看速度
# 测试集 batch 是否要有随机性 => 最好是不要，方便对比不同模型的效果


def hificodec_decode(hificodec, acoustic_token, rescale=True):
    """
    acoustic_token: shape [B, Nq, T]
    """
    # hificodec decode 需要 [B, T, Nq]
    # [B, Nq, T] -> [B, T, Nq]
    acoustic_token = acoustic_token.transpose(1, 2)
    # VQVAE.forward()
    wav = hificodec(acoustic_token)
    wav = wav.detach().squeeze().cpu().numpy()
    limit = 0.99
    if rescale:
        mx = np.abs(wav).max()
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clip(-limit, limit)
    return wav


def get_batch(acoustic_data, semantic_data, index, num_quant=4):
    '''
    一条数据构成一个 batch
    (1) 若总长度大于 6s, 前 3s 为 prompt, 剩余为 target 
    (2) 若总长度小于 6s, 则 1/2 分给 prompt, 剩余为 target 
    (3) target 最多为 10s
    '''
    hz = 50

    item_name = semantic_data['item_name'][index]
    semantic_str = semantic_data['semantic_audio'][index]
    # shape: (1, T)
    semantic_tokens = torch.tensor(
        [int(idx) for idx in semantic_str.split(' ')]).unsqueeze(0)
    try:
        acoustic_str = acoustic_data[item_name]
    except Exception:
        return None
    # shape (4, T)
    # acoustic_tokens 的 T 与 semantic_tokens 的 T 可能有误差
    acoustic_tokens = acoustic_str[:num_quant, ...]
    if acoustic_tokens.shape[1] > 6 * hz:
        prompt_len = 3 * hz
    else:
        prompt_len = acoustic_tokens.shape[1] // 2

    prompt_acoustic_tokens = acoustic_tokens[:, :prompt_len]
    prompt_semantic_tokens = semantic_tokens[:, :prompt_len]
    target_semantic_tokens = semantic_tokens[:, prompt_len:prompt_len + 10 * hz]
    prompt_semantic_tokens = prompt_semantic_tokens.cuda()
    target_semantic_tokens = target_semantic_tokens.cuda()
    prompt_acoustic_tokens = prompt_acoustic_tokens.cuda()

    real_acoustic_tokens = acoustic_tokens[:, prompt_len:prompt_len + 10 * hz]
    real_acoustic_tokens = real_acoustic_tokens.cuda()

    # 用 False 指示有值的位置, shape (1, T)
    x_mask = torch.zeros((1, real_acoustic_tokens.shape[-1])).bool().cuda()

    samples = {}
    # pseudo batch
    samples['prompt_semantics'] = prompt_semantic_tokens.unsqueeze(0)
    samples['target_semantics'] = target_semantic_tokens.unsqueeze(0)
    samples['prompt_acoustics'] = prompt_acoustic_tokens.unsqueeze(0)
    samples['target_acoustics'] = real_acoustic_tokens.unsqueeze(0)
    samples['x_mask'] = x_mask
    return samples


def evaluate(args):
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

    acoustic_data = torch.load(args.test_acoustic_path)
    semantic_data = pd.read_csv(args.test_semantic_path, delimiter='\t')
    num_quant = 4
    sample_rate = 16000

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, utt_id in enumerate(semantic_data['item_name'][:1]):
        # 需要处理 item_name 不在 acoustic_data 中的情况
        batch = get_batch(
            acoustic_data, semantic_data, index, num_quant=num_quant)
        # some wrong with this index od data
        if batch is None:
            continue

        with torch.no_grad():
            start = time.time()
            model_out = soundstorm.infer_one(batch)
            end = time.time()
            print("infer time:", end - start)

        content = model_out['token_pred']
        # shape (B, Nq x T) -> (B, Nq, T)
        codes = content.reshape(content.shape[0], num_quant, -1)
        wav_gen = hificodec_decode(hificodec, codes)
        wav_gt = hificodec_decode(hificodec, batch['target_acoustics'])

        sf.write(output_dir / (utt_id + ".wav"), wav_gen, sample_rate)
        sf.write(output_dir / (utt_id + "_real.wav"), wav_gt, sample_rate)


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
        '--test_semantic_path',
        type=str,
        default='dump/test/semantic_token.tsv')
    parser.add_argument(
        '--test_acoustic_path',
        type=str,
        default='dump/test/acoustic_token/hificodec.pth')

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
    evaluate(args)


if __name__ == "__main__":
    main()
