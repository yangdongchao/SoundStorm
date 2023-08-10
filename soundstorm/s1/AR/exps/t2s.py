# text to semantic
import argparse
import os
import re
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import whisper
from soundstorm.s1.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from soundstorm.s1.AR.text_processing.phonemizer import GruutPhonemizer
from soundstorm.s2.models.hubert.semantic_tokenizer import SemanticTokenizer
from soundstorm.utils.io import load_yaml_config


def get_batch(text, phonemizer):
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


def get_prompt(prompt_wav_path, asr_model, phonemizer, semantic_tokenizer):
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


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="Run SoundStorm AR S1 model for input text file")

    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/default.yaml',
        help='path of config file')

    parser.add_argument(
        "--text_file",
        type=str,
        help="text file to be convert to semantic tokens, a 'utt_id sentence' pair per line."
    )

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='exp/default/ckpt/epoch=99-step=49000.ckpt',
        help='Checkpoint file of SoundStorm AR S1 model.')

    parser.add_argument(
        '--prompt_wav_path',
        type=str,
        default=None,
        help='extract prompt semantic and prompt phonemes from prompt wav')

    # to get semantic tokens from prompt_wav
    parser.add_argument("--hubert_path", type=str, default=None)
    parser.add_argument("--quantizer_path", type=str, default=None)

    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = load_yaml_config(args.config_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hz = 50
    max_sec = config['data']['max_sec']

    # get models
    t2s_model = Text2SemanticLightningModule.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, config=config)
    t2s_model.cuda()
    t2s_model.eval()

    phonemizer: GruutPhonemizer = GruutPhonemizer(language='en-us')

    # models for prompt
    asr_model = whisper.load_model("tiny.en")

    semantic_tokenizer = SemanticTokenizer(
        hubert_path=args.hubert_path,
        quantizer_path=args.quantizer_path,
        duplicate=True)

    prompt_result = get_prompt(
        prompt_wav_path=args.prompt_wav_path,
        asr_model=asr_model,
        phonemizer=phonemizer,
        semantic_tokenizer=semantic_tokenizer)

    # zero prompt => 输出的 semantic 包含的内容是对的但是音色是乱的
    # (B, 1)
    # prompt = torch.ones(
    #     batch['phoneme_ids'].size(0), 1, dtype=torch.int32) * 0

    prompt = prompt_result['prompt_semantic_tokens']
    prompt_phoneme_ids_len = prompt_result['prompt_phoneme_ids_len']
    prompt_phoneme_ids = prompt_result['prompt_phoneme_ids']

    sentences = []
    with open(args.text_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip() != "":
                items = re.split(r"\s+", line.strip(), 1)
                utt_id = items[0]
                sentence = " ".join(items[1:])
            sentences.append((utt_id, sentence))
    semantic_data = [['item_name', 'semantic_audio']]
    for utt_id, sentence in sentences[1:]:
        # 需要自己构造伪 batch 输入给模型
        batch = get_batch(sentence, phonemizer)
        # prompt 和真正的输入拼接
        all_phoneme_ids = torch.cat(
            [prompt_phoneme_ids, batch['phoneme_ids']], dim=1)
        # 或者可以直接求 all_phoneme_ids 的 shape[-1]
        all_phoneme_len = prompt_phoneme_ids_len + batch['phoneme_ids_len']
        st = time.time()
        with torch.no_grad():
            pred_semantic = t2s_model.model.infer(
                all_phoneme_ids.cuda(),
                all_phoneme_len.cuda(),
                prompt.cuda(),
                top_k=config['inference']['top_k'],
                early_stop_num=hz * max_sec)
        print(f'{time.time() - st} sec used in T2S')

        # 删除 prompt 对应的部分
        prompt_len = prompt.shape[-1]
        pred_semantic = pred_semantic[:, prompt_len:]

        # bs = 1
        pred_semantic = pred_semantic[0]
        semantic_token = pred_semantic.detach().cpu().numpy().tolist()
        semantic_token_str = ' '.join(str(x) for x in semantic_token)
        semantic_data.append([utt_id, semantic_token_str])

        delimiter = '\t'
        filename = output_dir / f'{utt_id}_p_{prompt_result["prompt_name"]}_semantic_token.tsv'
        with open(filename, 'w', encoding='utf-8') as writer:
            for row in semantic_data:
                line = delimiter.join(row)
                writer.write(line + '\n')
        # clean semantic token for next setence
        semantic_data = [['item_name', 'semantic_audio']]


if __name__ == "__main__":
    main()
