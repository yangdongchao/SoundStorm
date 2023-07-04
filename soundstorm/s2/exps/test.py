import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from academicodec.models.hificodec.vqvae import VQVAE
from soundstorm.s2.data.semantic_dataset import pad_2D
from soundstorm.s2.models.dalle_wav.build import build_model
from soundstorm.s2.utils.io import load_yaml_config

# 每一条构成一个 batch 过一遍模型
# 是单条推理还是凑 batch 推理？=> 可以实现两种分别看速度
# 测试集 batch 是否要有随机性 => 最好是不要，方便对比不同模型的效果

semantic_token_nums = 1000
prompt_semantic_end_id = semantic_token_nums + 1
target_semantic_end_id = semantic_token_nums + 3
acoustic_token_nums = 1024
prompt_acoustic_eos = acoustic_token_nums
target_acoustic_eos = acoustic_token_nums + 1


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


def get_one_sample(acoustic_data, semantic_data, index, num_quant):
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
    prompt_semantic_tokens = prompt_semantic_tokens
    target_semantic_tokens = target_semantic_tokens
    prompt_acoustic_tokens = prompt_acoustic_tokens

    target_acoustics_tokens = acoustic_tokens[:, prompt_len:prompt_len + 10 *
                                              hz]
    target_acoustics_tokens = target_acoustics_tokens

    result = {}
    result['prompt_acoustic_tokens'] = prompt_acoustic_tokens
    result['prompt_semantic_tokens'] = prompt_semantic_tokens
    result['target_semantic_tokens'] = target_semantic_tokens
    result['target_acoustics_tokens'] = target_acoustics_tokens

    return result


# one wav per batch
def get_batch(acoustic_data, semantic_data, index, num_quant=4):
    result = get_one_sample(
        acoustic_data=acoustic_data,
        semantic_data=semantic_data,
        index=index,
        num_quant=num_quant)
    prompt_acoustic_tokens = result['prompt_acoustic_tokens']
    prompt_semantic_tokens = result['prompt_semantic_tokens']
    target_semantic_tokens = result['target_semantic_tokens']
    target_acoustics_tokens = result['target_acoustics_tokens']
    # 用 False 指示有值的位置, shape (1, T)
    x_mask = torch.zeros((1, target_acoustics_tokens.shape[-1])).bool()
    samples = {}
    # pseudo batch
    samples['prompt_semantics'] = prompt_semantic_tokens.unsqueeze(0)
    samples['target_semantics'] = target_semantic_tokens.unsqueeze(0)
    samples['prompt_acoustics'] = prompt_acoustic_tokens.unsqueeze(0)
    samples['target_acoustics'] = target_acoustics_tokens.unsqueeze(0)
    samples['x_mask'] = x_mask
    return samples


def get_big_batch(acoustic_data, semantic_data, index_list, num_quant=4):
    tmp_prompt_semantics = []
    tmp_target_semantics = []
    tmp_prompt_acoustics = []
    tmp_target_acoustics = []
    for index in index_list:
        result = get_one_sample(
            acoustic_data=acoustic_data,
            semantic_data=semantic_data,
            index=index,
            num_quant=num_quant)

        prompt_semantic = result['prompt_semantic_tokens']
        target_semantic = result['target_semantic_tokens']
        prompt_acoustic = result['prompt_acoustic_tokens']
        target_acoustic = result['target_acoustics_tokens']

        # 凑 batch
        tmp_prompt_semantics.append(prompt_semantic)
        tmp_target_semantics.append(target_semantic)
        tmp_prompt_acoustics.append(prompt_acoustic)
        tmp_target_acoustics.append(target_acoustic)

    # 一个 batch 里面按照最长的补 0
    prompt_semantics = pad_2D(tmp_prompt_semantics, prompt_semantic_end_id)
    target_semantics = pad_2D(tmp_target_semantics, target_semantic_end_id)
    prompt_acoustics = pad_2D(tmp_prompt_acoustics, prompt_acoustic_eos)
    # 用 1025 补零
    target_acoustics = pad_2D(tmp_target_acoustics, target_acoustic_eos)
    # mask 住 target_acoustics 的补 0 部分
    x_mask = (target_acoustics == target_acoustic_eos)
    new_samples = {}
    # (B, 1, T), B, T 动态
    new_samples['prompt_semantics'] = torch.from_numpy(prompt_semantics)
    new_samples['target_semantics'] = torch.from_numpy(target_semantics)
    new_samples['prompt_acoustics'] = torch.from_numpy(prompt_acoustics)
    # (B, 4, T), B, T 动态
    new_samples['target_acoustics'] = torch.from_numpy(target_acoustics)
    # (B, T)
    new_samples['x_mask'] = torch.from_numpy(x_mask[:, 0, :])

    return new_samples


# evaluate one wav per batch
def evaluate(args, hificodec, soundstorm):
    num_quant = 4
    sample_rate = 16000
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acoustic_data = torch.load(args.test_acoustic_path)
    semantic_data = pd.read_csv(args.test_semantic_path, delimiter='\t')

    for index, utt_id in enumerate(semantic_data['item_name'][:20]):
        # 需要处理 item_name 不在 acoustic_data 中的情况
        batch = get_batch(
            acoustic_data, semantic_data, index, num_quant=num_quant)
        batch = move_tensors_to_cuda(batch)
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

        sf.write(output_dir / (utt_id + "_bs1.wav"), wav_gen, sample_rate)
        sf.write(output_dir / (utt_id + "_real_bs1.wav"), wav_gt, sample_rate)


# evaluate batch
def evaluate_batch(args, hificodec, soundstorm, batch_size=2):
    # 按照顺序读取测试集，若干调音频组成一个 batch
    num_quant = 4
    sample_rate = 16000
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    acoustic_data = torch.load(args.test_acoustic_path)
    semantic_data = pd.read_csv(args.test_semantic_path, delimiter='\t')

    # split data into n batch with batch_size
    utt_id_list = semantic_data['item_name'].tolist()
    utt_id_lists = [
        utt_id_list[i:i + batch_size]
        for i in range(0, len(utt_id_list), batch_size)
    ]

    all_indexs = list(range(0, len(utt_id_list)))
    index_lists = [
        all_indexs[i:i + batch_size]
        for i in range(0, len(all_indexs), batch_size)
    ]

    for i, index_list in enumerate(index_lists):
        utt_ids = utt_id_lists[i]
        batch = get_big_batch(
            acoustic_data, semantic_data, index_list, num_quant=4)
        batch = move_tensors_to_cuda(batch)
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
        # to clip wav
        eos_indexs = []
        for j in range(0, codes.shape[0]):
            utt_id = utt_ids[j]
            # to clip wav with mask
            codes_gt = batch['target_acoustics'][j]
            codes_gt_0_list = codes_gt[0].cpu().numpy().tolist()
            eos_index = codes_gt_0_list.index(
                target_acoustic_eos
            ) if target_acoustic_eos in codes_gt_0_list else -1
            eos_indexs.append(eos_index)
            # pseudo batch
            wav_gt = hificodec_decode(hificodec,
                                      codes_gt.unsqueeze(0)[:, :, :eos_index])
            wav_gen = hificodec_decode(hificodec,
                                       codes[j].unsqueeze(0)[:, :, :eos_index])

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
    # cost 14s for a 10s target
    evaluate(args, hificodec, soundstorm)
    # evaluate_batch(args, hificodec, soundstorm, 2)


if __name__ == "__main__":
    main()
