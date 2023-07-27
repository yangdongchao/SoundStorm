# merge soundstorm/s1/AR/exps/t2s.py and soundstorm/s2/exps/synthesize.py
import argparse
import os
import time
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
    phoneme_ids = phonemizer.transform(phoneme)
    phoneme_ids_len = len(phoneme_ids)
    phoneme_ids = np.array(phoneme_ids)
    # add batch axis here
    phoneme_ids = torch.tensor(phoneme_ids).unsqueeze(0)
    phoneme_ids_len = torch.tensor([phoneme_ids_len])
    print("phoneme:", phoneme)
    batch = {
        # torch.Tensor (B, max_phoneme_length) 
        "phoneme_ids": phoneme_ids,
        # torch.Tensor (B)
        "phoneme_ids_len": phoneme_ids_len
    }
    return batch


def get_S1_prompt(prompt_wav_path, asr_model, phonemizer, semantic_tokenizer):
    sample_rate = 16000
    # to get prompt
    prompt_name = os.path.basename(prompt_wav_path).split('.')[0]
    wav, _ = librosa.load(prompt_wav_path, sr=sample_rate)
    # 取末尾 3s, 但是不包含最后 0.1s 防止 AR S1 infer 提前停止
    wav = wav[-sample_rate * 3:-int(sample_rate * 0.1)]
    # wav 需要挪出末尾的静音否则也可能提前停住
    prompt_text = asr_model.transcribe(wav)["text"]
    # 移除最后的句点, 防止 AR S1 infer 提前停止, 加了句点可能会有停顿
    prompt_text = prompt_text.replace(".", "")
    prompt_phoneme = phonemizer.phonemize(prompt_text, espeak=False)
    prompt_phoneme_ids = phonemizer.transform(prompt_phoneme)
    prompt_phoneme_ids_len = len(prompt_phoneme_ids)
    # get prompt_semantic
    # (T) -> (1, T)
    wav = torch.tensor(wav).unsqueeze(0)
    wav = wav.cuda()
    # (1, T)
    prompt_semantic_tokens = semantic_tokenizer.tokenize(wav).to(torch.int32)
    prompt_phoneme_ids = torch.tensor(prompt_phoneme_ids).unsqueeze(0)
    prompt_phoneme_ids_len = torch.tensor([prompt_phoneme_ids_len])

    result = {
        'prompt_name': prompt_name,
        'prompt_phoneme_ids': prompt_phoneme_ids,
        'prompt_semantic_tokens': prompt_semantic_tokens,
        'prompt_phoneme_ids_len': prompt_phoneme_ids_len
    }

    return result


# one wav per batch
def get_S2_batch(prompt_semantic_tokens,
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


def evaluate(args, S1_model, S2_model, semantic_tokenizer, asr_model, hificodec,
             phonemizer):
    num_quant = 4
    sample_rate = 16000
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # get prompt_semantic and prompt_acoustic from prompt_wav

    prompt_name = os.path.basename(args.prompt_wav_path).split('.')[0]
    wav, _ = librosa.load(args.prompt_wav_path, sr=16000)
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
    '''
    # get target
    # 保留 item_name 前导 0
    target_semantic_data = pd.read_csv(
        args.target_semantic_path, delimiter='\t', dtype=str)
    target_name = target_semantic_data['item_name'][0]
    target_semantic_str = target_semantic_data['semantic_audio'][0]
    # shape: (1, T)
    target_semantic_tokens = torch.tensor(
        [int(idx) for idx in target_semantic_str.split(' ')]).unsqueeze(0)
    '''
    # get target semnatic from prompt wav and text

    batch = get_S2_batch(
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

    sf.write(output_dir /
             ("s_" + prompt_name + "_t_" + target_name + "_ok5.wav"), wav_gen,
             sample_rate)


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
    s2_model = build_model(S2_config)
    s2_model.load_state_dict(S2_ckpt["model"])
    s2_model.eval()
    s2_model.cuda()

    # get prompt_semantic from prompt_wav
    semantic_tokenizer = SemanticTokenizer(
        hubert_path=args.hubert_path,
        quantizer_path=args.quantizer_path,
        duplicate=True)

    phonemizer = GruutPhonemizer(language='en-us')
    asr_model = whisper.load_model("tiny")

    # cost 14s for a 10s target
    evaluate(
        args,
        S1_model=S1_model,
        S2_model=S2_model,
        semantic_tokenizer=semantic_tokenizer,
        asr_model=asr_model,
        hificodec=hificodec,
        phonemizer=phonemizer)


if __name__ == "__main__":
    main()
