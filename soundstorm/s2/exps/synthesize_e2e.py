# merge soundstorm/s1/AR/exps/t2s.py and soundstorm/s2/exps/synthesize.py
import argparse
import os
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from academicodec.models.hificodec.vqvae import VQVAE
from soundstorm.s1.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from soundstorm.s1.AR.text_processing.phonemizer import GruutPhonemizer
from soundstorm.s2.models.dalle_wav.build import build_model
from soundstorm.s2.models.hubert.semantic_tokenizer import SemanticTokenizer
from soundstorm.utils.io import load_yaml_config
from timer import timer

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


def get_S1_batch(text, phonemizer):
    # phoneme_ids 和 phoneme_ids_len 是需要的
    phoneme = phonemizer.phonemize(text, espeak=False)
    # print("phoneme:", phoneme)
    phoneme = " " + phoneme
    phoneme_ids = phonemizer.transform(phoneme)
    phoneme_ids_len = len(phoneme_ids)
    phoneme_ids = np.array(phoneme_ids)
    # add batch axis here
    phoneme_ids = torch.tensor(phoneme_ids).unsqueeze(0)
    phoneme_ids_len = torch.tensor([phoneme_ids_len])
    # print("phoneme_ids:", phoneme_ids)
    batch = {
        # torch.Tensor (B, max_phoneme_length) 
        "phoneme_ids": phoneme_ids,
        # torch.Tensor (B)
        "phoneme_ids_len": phoneme_ids_len
    }
    return batch


def get_S1_prompt(wav, asr_model, phonemizer, semantic_tokenizer, sil_token):
    # wav 需要挪出末尾的静音否则也可能提前停住
    prompt_text = asr_model.transcribe(wav)["text"]
    # print("prompt_text:", prompt_text)
    # 移除最后的句点, 防止 AR S1 infer 提前停止, 加了句点可能会有停顿或者噪音
    # prompt_text = prompt_text.replace(".", "")
    prompt_phoneme = phonemizer.phonemize(prompt_text, espeak=False)
    # 移除 prompt_phoneme 末尾的 . 但是保留空格, 因为直接移除 prompt_text 会导致 prompt_phoneme 末尾少一个空格
    prompt_phoneme = prompt_phoneme.rstrip(".")
    # print("prompt_phoneme:", prompt_phoneme)
    # 末尾不是空格的则加一个空格
    if not prompt_phoneme.endswith(' '):
        prompt_phoneme = prompt_phoneme + ' '
    # 16 表示空格
    prompt_phoneme_ids = phonemizer.transform(prompt_phoneme)
    # print("prompt_phoneme_ids:", prompt_phoneme_ids)
    prompt_phoneme_ids_len = len(prompt_phoneme_ids)
    # get prompt_semantic
    # (T) -> (1, T)
    wav = torch.tensor(wav).unsqueeze(0)
    wav = wav.cuda()
    # (1, T)
    prompt_semantic_tokens = semantic_tokenizer.tokenize(wav).to(torch.int32)
    # print("prompt_semantic_tokens:", prompt_semantic_tokens)
    # remove sil token in the end
    while prompt_semantic_tokens[0][-1] == sil_token:
        prompt_semantic_tokens = prompt_semantic_tokens[..., :-1]
        # print("prompt_semantic_tokens:", prompt_semantic_tokens)
    prompt_phoneme_ids = torch.tensor(prompt_phoneme_ids).unsqueeze(0)
    prompt_phoneme_ids_len = torch.tensor([prompt_phoneme_ids_len])

    result = {
        'prompt_phoneme_ids': prompt_phoneme_ids,
        'prompt_semantic_tokens': prompt_semantic_tokens,
        'prompt_phoneme_ids_len': prompt_phoneme_ids_len
    }

    return result


def get_S2_prompt(wav, semantic_tokenizer, hificodec):
    wav = torch.tensor(wav).unsqueeze(0)
    wav = wav.cuda()
    # get prompt_semantic
    # (1, T)
    prompt_semantic_tokens = semantic_tokenizer.tokenize(wav)

    # get prompt_acoustic
    # (1, T, 4)
    acoustic_token = hificodec.encode(wav)
    # trans acoustic_token.shape to (Nq, T)
    prompt_acoustic_tokens = acoustic_token.squeeze(0).transpose(0, 1)

    result = {
        'prompt_semantic_tokens': prompt_semantic_tokens,
        'prompt_acoustic_tokens': prompt_acoustic_tokens
    }

    return result


# one wav per batch
def get_S2_batch(prompt_semantic_tokens,
                 prompt_acoustic_tokens,
                 target_semantic_tokens,
                 num_quant: int=4,
                 hz: int=50,
                 max_prompt_sec: int=3,
                 max_target_sec: int=10):
    # transformer_utils.py 里面最大是 20, pad 了一个  stop token, 所以这里最大是 19
    # 但是训练时最多是 10s, 所以超过 10s 的无法合成出来

    # prompt 最多为 3s
    if prompt_acoustic_tokens.shape[1] > 2 * max_prompt_sec * hz:
        prompt_len = max_prompt_sec * hz
    else:
        prompt_len = prompt_acoustic_tokens.shape[1] // 2

    prompt_semantic_tokens = prompt_semantic_tokens[:, :prompt_len]
    prompt_acoustic_tokens = prompt_acoustic_tokens[:, :prompt_len]
    # target 最多为 10s
    target_semantic_tokens = target_semantic_tokens[:, :max_target_sec * hz]
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


def get_prompt_wav(prompt_wav_path, sample_rate):
    print("----------------------------------------")
    prompt_wav, _ = librosa.load(prompt_wav_path, sr=sample_rate)
    # 取末尾 3s, 但是不包含最后 0.1s 防止 AR S1 infer 提前停止
    print("prompt_wav.shape:", prompt_wav.shape)
    # 减小 top_db 数值，在末尾有长拖音时可以适当裁剪，减少 early stop 的概率
    prompt_wav, index = librosa.effects.trim(
        prompt_wav, top_db=15, frame_length=256, hop_length=64)
    print("prompt_wav.shape after trim:", prompt_wav.shape)
    # 有可能截取的地方刚好是一个字，所以 asr 的结果和 wav 无法对齐
    # 会裁掉 2 个 semantic token
    prompt_wav = prompt_wav[-3 * sample_rate:int(-0.05 * sample_rate)]
    # prompt_wav = prompt_wav[-3 * sample_rate:]
    print("prompt_wav.shape after slice:", prompt_wav.shape)
    prompt_wav, index = librosa.effects.trim(
        prompt_wav, top_db=15, frame_length=256, hop_length=64)
    print("prompt_wav.shape after trim:", prompt_wav.shape)
    # add small sil, sil 太大会导致 S1 无法推理或者开头结果被忽略,
    # 但是没有 sil 可能导致 S2 结果有噪音
    # np_zero = np.array([0] * int(sample_rate / 200), dtype='float32')
    # prompt_wav = np.concatenate([prompt_wav, np_zero])
    print("prompt_wav.shape after add sil:", prompt_wav.shape)
    print("----------------------------------------")
    return prompt_wav


def evaluate(args,
             S1_model,
             S2_model,
             semantic_tokenizer,
             asr_model,
             hificodec,
             phonemizer,
             S1_max_sec: int=20,
             S1_top_k: int=-100,
             S2_max_sec: int=10,
             S2_max_prompt_sec: int=3,
             S2_max_target_sec: int=10):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_quant = 4
    sample_rate = 16000
    hz = 50
    # 设置为 0.5 时重复情况较为严重, 表现为一个音拖得很长
    # 1.1 音色有点不像了
    S1_temperature = 1.05

    sentences = []
    with open(args.text_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip() != "":
                items = re.split(r"\s+", line.strip(), 1)
                utt_id = items[0]
                sentence = " ".join(items[1:])
            sentences.append((utt_id, sentence))

    # to get prompt
    prompt_name = os.path.basename(args.prompt_wav_path).split('.')[0]
    prompt_wav = get_prompt_wav(args.prompt_wav_path, sample_rate)

    # get prompt for S1
    S1_prompt_result = get_S1_prompt(
        wav=prompt_wav,
        asr_model=asr_model,
        phonemizer=phonemizer,
        semantic_tokenizer=semantic_tokenizer,
        sil_token=args.sil_token)

    S1_prompt = S1_prompt_result['prompt_semantic_tokens']

    S1_prompt_phoneme_ids_len = S1_prompt_result['prompt_phoneme_ids_len']
    S1_prompt_phoneme_ids = S1_prompt_result['prompt_phoneme_ids']

    # get prompt_semantic and prompt_acoustic from prompt_wav
    S2_prompt_result = get_S2_prompt(
        wav=prompt_wav,
        semantic_tokenizer=semantic_tokenizer,
        hificodec=hificodec)
    # prompt input for S2
    prompt_semantic_tokens = S2_prompt_result['prompt_semantic_tokens']
    prompt_acoustic_tokens = S2_prompt_result['prompt_acoustic_tokens']

    N = 0
    T = 0

    # 遍历 utt_id
    for utt_id, sentence in sentences:
        with timer() as t:
            S1_batch = get_S1_batch(sentence, phonemizer)
            # prompt 和真正的输入拼接
            S1_all_phoneme_ids = torch.cat(
                [S1_prompt_phoneme_ids, S1_batch['phoneme_ids']], dim=1)
            S1_all_phoneme_len = S1_prompt_phoneme_ids_len + S1_batch[
                'phoneme_ids_len']
            S1_st = t.elapse
            with torch.no_grad():
                S1_pred_semantic = S1_model.model.infer(
                    S1_all_phoneme_ids.cuda(),
                    S1_all_phoneme_len.cuda(),
                    S1_prompt.cuda(),
                    top_k=S1_top_k,
                    early_stop_num=hz * S1_max_sec,
                    temperature=S1_temperature)
            S1_end = t.elapse

            # 删除 prompt 对应的部分
            S1_prompt_len = S1_prompt.shape[-1]
            # print("S1_prompt_len:", S1_prompt_len)
            # (1, T)
            # target phone 开头有多余的 " ", 所以这里可以多截掉大约 3 个
            S1_pred_semantic = S1_pred_semantic[:, S1_prompt_len + 3:]

            # get target semnatic from prompt wav and text
            S2_batch = get_S2_batch(
                prompt_semantic_tokens=prompt_semantic_tokens,
                prompt_acoustic_tokens=prompt_acoustic_tokens,
                target_semantic_tokens=S1_pred_semantic,
                num_quant=num_quant,
                hz=hz,
                max_prompt_sec=S2_max_prompt_sec,
                max_target_sec=S2_max_target_sec)

            S2_batch = move_tensors_to_cuda(S2_batch)

            S2_st = t.elapse
            with torch.no_grad():
                model_out = S2_model.infer_one(S2_batch)
            S2_end = t.elapse

            content = model_out['token_pred']
            # shape (B, Nq x T) -> (B, Nq, T)
            codes = content.reshape(content.shape[0], num_quant, -1)
            wav_gen = hificodec_decode(hificodec, codes)

        N += wav_gen.size
        T += t.elapse
        speed = wav_gen.size / t.elapse
        rtf = sample_rate / speed
        print(
            f"{utt_id},  wave: {wav_gen.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
        print(f'{S1_end - S1_st} sec used in T2S')
        print(f'{S2_end - S2_st} sec used in S2A')

        sf.write(
            output_dir /
            f"S1_topk_{S1_top_k}_temperature_{S1_temperature}_s_{prompt_name}_t_{utt_id}.wav",
            np.concatenate([prompt_wav, wav_gen]), sample_rate)
    print(f"generation speed: {N / T}Hz, RTF: {sample_rate / (N / T) }")


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(description="Run SoundStorm for test set.")

    parser.add_argument(
        '--S1_config_file',
        type=str,
        default='conf/default.yaml',
        help='path of S1 (text to semantic) config file')

    parser.add_argument(
        '--S1_ckpt_path',
        type=str,
        help='Checkpoint file of SoundStorm S1 model.')

    parser.add_argument(
        '--S2_config_file',
        type=str,
        default='conf/default.yaml',
        help='path of S2 (semantic to acoustic) config file')

    parser.add_argument(
        '--S2_ckpt_path',
        type=str,
        default='exp/default/checkpoint/last.pth',
        help='Checkpoint file of SoundStorm S2 model.')

    parser.add_argument(
        '--prompt_wav_path',
        type=str,
        default=None,
        help='extract prompt semantic and prompt phonemes from prompt wav')
    parser.add_argument(
        '--sil_token',
        type=int,
        help='sil token of your semantic token, 193 for 500 bin')

    # to get semantic tokens from prompt_wav
    parser.add_argument("--hubert_path", type=str, default=None)
    parser.add_argument("--quantizer_path", type=str, default=None)

    parser.add_argument(
        "--text_file",
        type=str,
        help="text file to be convert to wav, a 'utt_id sentence' pair per line."
    )

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

    # get models
    # get S1 model

    S1_config = load_yaml_config(args.S1_config_file)
    S1_model = Text2SemanticLightningModule.load_from_checkpoint(
        checkpoint_path=args.S1_ckpt_path, config=S1_config)
    S1_model.cuda()
    S1_model.eval()

    # get S2 model
    S2_config = load_yaml_config(args.S2_config_file)
    S2_ckpt = torch.load(args.S2_ckpt_path, map_location="cpu")
    S2_model = build_model(S2_config)
    S2_model.load_state_dict(S2_ckpt["model"])
    S2_model.eval()
    S2_model.cuda()

    # get prompt_semantic from prompt_wav
    semantic_tokenizer = SemanticTokenizer(
        hubert_path=args.hubert_path,
        quantizer_path=args.quantizer_path,
        duplicate=True)

    phonemizer = GruutPhonemizer(language='en-us')
    asr_model = whisper.load_model("tiny.en")
    S2_max_prompt_sec = S2_config['dataloader']['train_datasets'][0][
        'params'].get('max_prompt_sec', 3)
    S2_max_target_sec = S2_config['dataloader']['train_datasets'][0][
        'params'].get('max_target_sec', 10)

    # cost 14s for a 10s target
    evaluate(
        args,
        S1_model=S1_model,
        S2_model=S2_model,
        semantic_tokenizer=semantic_tokenizer,
        asr_model=asr_model,
        hificodec=hificodec,
        phonemizer=phonemizer,
        S1_max_sec=S1_config['data']['max_sec'],
        S1_top_k=S1_config['inference']['top_k'],
        S2_max_prompt_sec=S2_max_prompt_sec,
        S2_max_target_sec=S2_max_target_sec)


if __name__ == "__main__":
    main()
