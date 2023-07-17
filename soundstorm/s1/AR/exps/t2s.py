# text to semantic
import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from soundstorm.s1.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from soundstorm.s1.AR.text_processing.phonemizer import GruutPhonemizer
"""
这里 to 转写有点问题
>>> a['1001_134708_000013_000000']
'fɚ mæn əv ju , jʊɹ kɛɹɪktɚɪstɪk ɹeɪs , hɪɹ meɪ hi hɑɹdi , swit , dʒaɪɡæntɪk ɡɹoʊ , hɪɹ taʊɚ pɹəpɔɹʃənɪt tə neɪtʃɚ , hɪɹ klaɪm ðə væst pjʊɹ speɪsɪz ʌŋkənfaɪnd , ʌntʃɛkt baɪ wɔl ɔɹ ɹuf , hɪɹ læf wɪθ stɔɹm ɔɹ sʌn , hɪɹ dʒɔɪ , hɪɹ peɪʃəntli ɪnjʊɹ , hɪɹ hid hɪmsɛlf , ʌnfoʊld hɪmsɛlf ,  nɑt ʌðɚz  fɔɹmjuləz hid ,  hɪɹ fɪl hɪz taɪm , tə duli fɔl , tə eɪd , ʌnɹɛkt æt læst , tə dɪsəpɪɹ , tə sɚv .'
For man of you, your characteristic race, Here may he hardy, sweet, gigantic grow, here tower proportionate to Nature, Here climb the vast pure spaces unconfined, uncheck'd by wall or roof, Here laugh with storm or sun, here joy, here patiently inure, Here heed himself, unfold himself, (not others' formulas heed,) here fill his time, To duly fall, to aid, unreck'd at last, To disappear, to serve.

{'ids': [0], 'phoneme_ids': tensor([[ 48,  85,  16,  55,  72,  56,  16,  83,  64,  16,  52,  63,  16,   3,
          16,  52, 135, 123,  16,  53,  86, 123, 102,  53,  62,  85, 102,  61,
          62, 102,  53,  16, 123,  47, 102,  61,  16,   3,  16,  50, 102, 123,
          16,  55,  47, 102,  16,  50,  51,  16,  50,  69, 123,  46,  51,  16,
           3,  16,  61,  65,  51,  62,  16,   3,  16,  46, 147,  43, 102,  92,
          72,  56,  62, 102,  53,  16,  92, 123,  57, 135,  16,   3,  16,  50,
         102, 123,  16,  62,  43, 135,  85,  16,  58, 123,  83,  58,  76, 123,
         131,  83,  56, 102,  62,  16,  62,  83,  16,  56,  47, 102,  62, 131,
          85,  16,   3,  16,  50, 102, 123,  16,  53,  54,  43, 102,  55,  16,
          81,  83,  16,  64,  72,  61,  62,  16,  58,  52, 135, 123,  16,  61,
          58,  47, 102,  61, 102,  68,  16, 138, 112,  53,  83,  56,  48,  43,
         102,  56,  46,  16,   3,  16, 138,  56,  62, 131,  86,  53,  62,  16,
          44,  43, 102,  16,  65,  76,  54,  16,  76, 123,  16, 123,  63,  48,
          16,   3,  16,  50, 102, 123,  16,  54,  72,  48,  16,  65, 102, 119,
          16,  61,  62,  76, 123,  55,  16,  76, 123,  16,  61, 138,  56,  16,
           3,  16,  50, 102, 123,  16,  46, 147,  76, 102,  16,   3,  16,  50,
         102, 123,  16,  58,  47, 102, 131,  83,  56,  62,  54,  51,  16, 102,
          56,  52, 135, 123,  16,   3,  16,  50, 102, 123,  16,  50,  51,  46,
          16,  50, 102,  55,  61,  86,  54,  48,  16,   3,  16, 138,  56,  48,
          57, 135,  54,  46,  16,  50, 102,  55,  61,  86,  54,  48,  16,   3,
          16,  16,  56,  69,  62,  16, 138,  81,  85,  68,  16,  16,  48,  76,
         123,  55,  52,  63,  54,  83,  68,  16,  50,  51,  46,  16,   3,  16,
          16,  50, 102, 123,  16,  48, 102,  54,  16,  50, 102,  68,  16,  62,
          43, 102,  55,  16,   3,  16,  62,  83,  16,  46,  63,  54,  51,  16,
          48,  76,  54,  16,   3,  16,  62,  83,  16,  47, 102,  46,  16,   3,
          16, 138,  56, 123,  86,  53,  62,  16,  72,  62,  16,  54,  72,  61,
          62,  16,   3,  16,  62,  83,  16,  46, 102,  61,  83,  58, 102, 123,
          16,   3,  16,  62,  83,  16,  61,  85,  64,  16,   4]]), 'phoneme_ids_len': tensor([389]), 'semantic_ids': tensor([[475, 475, 641,  ..., 984,  80, 244]]), 'semantic_ids_len': tensor([1845])}
prompt.shape: torch.Size([1, 150])
T2S Decoding EOS [150 -> 663]
33.47231197357178 sec used in T2S
"""
'''
batch: {'phoneme_ids': tensor([ 48,  85,  16,  55,  72,  56,  16,  83,  64,  16,  52,  63,  16,   3,
         16,  52, 135, 123,  16,  53,  86, 123, 102,  53,  62,  85, 102,  61,
         62, 102,  53,  16, 123,  47, 102,  61,  16,   3,  16,  50, 102, 123,
         16,  55,  47, 102,  16,  50,  51,  16,  50,  69, 123,  46,  51,  16,
          3,  16,  61,  65,  51,  62,  16,   3,  16,  46, 147,  43, 102,  92,
         72,  56,  62, 102,  53,  16,  92, 123,  57, 135,  16,   3,  16,  50,
        102, 123,  16,  62,  43, 135,  85,  16,  58, 123,  83,  58,  76, 123,
        131,  83,  56, 102,  62,  16,  62,  83,  16,  56,  47, 102,  62, 131,
         85,  16,   3,  16,  50, 102, 123,  16,  53,  54,  43, 102,  55,  16,
         81,  83,  16,  64,  72,  61,  62,  16,  58,  52, 135, 123,  16,  61,
         58,  47, 102,  61, 102,  68,  16, 138, 112,  53,  83,  56,  48,  43,
        102,  56,  46,  16,   3,  16, 138,  56,  62, 131,  86,  53,  62,  16,
         44,  43, 102,  16,  65,  76,  54,  16,  76, 123,  16, 123,  63,  48,
         16,   3,  16,  50, 102, 123,  16,  54,  72,  48,  16,  65, 102, 119,
         16,  61,  62,  76, 123,  55,  16,  76, 123,  16,  61, 138,  56,  16,
          3,  16,  50, 102, 123,  16,  46, 147,  76, 102,  16,   3,  16,  50,
        102, 123,  16,  58,  47, 102, 131,  83,  56,  62,  54,  51,  16, 102,
         56,  52, 135, 123,  16,   3,  16,  50, 102, 123,  16,  50,  51,  46,
         16,  50, 102,  55,  61,  86,  54,  48,  16,   3,  16, 138,  56,  48,
         57, 135,  54,  46,  16,  50, 102,  55,  61,  86,  54,  48,  16,   3,
         16,  16,  56,  69,  62,  16, 138,  81,  85,  68,  16,  16,  48,  76,
        123,  55,  52,  63,  54,  83,  68,  16,  50,  51,  46,  16,   3,  16,
         16,  50, 102, 123,  16,  48, 102,  54,  16,  50, 102,  68,  16,  62,
         43, 102,  55,  16,   3,  16,  62,  83,  16,  46,  63,  54,  51,  16,
         48,  76,  54,  16,   3,  16,  62,  83,  16,  47, 102,  46,  16,   3,
         16, 138,  56, 123,  86,  53,  62,  16,  72,  62,  16,  54,  72,  61,
         62,  16,   3,  16,  62,  83,  16,  46, 102,  61,  83,  58, 102, 123,
         16,   3,  16,  62,  83,  16,  61,  85,  64,  16,   4]), 'phoneme_ids_len': tensor([389])}
'''


def get_batch(text, phonemizer):
    # phoneme_ids 和 phoneme_ids_len 是需要的
    phoneme = phonemizer.phonemize(text, espeak=False)
    phoneme_ids = phonemizer.transform(phoneme)
    phoneme_ids_len = len(phoneme_ids)
    phoneme_ids = np.array(phoneme_ids)
    phoneme_ids = torch.tensor(phoneme_ids)
    phoneme_ids_lens = torch.tensor([phoneme_ids_len])
    batch = {
        # torch.Tensor (B, max_phoneme_length) 
        "phoneme_ids": phoneme_ids,
        # torch.Tensor (B)
        "phoneme_ids_len": phoneme_ids_lens
    }
    return batch


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

    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # get models
    t2s_model = Text2SemanticLightningModule.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, config=config)
    t2s_model.cuda()
    t2s_model.eval()

    phonemizer: GruutPhonemizer = GruutPhonemizer(language='en-us')

    sentences = []
    with open(args.text_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip() != "":
                items = re.split(r"\s+", line.strip(), 1)
                utt_id = items[0]
                sentence = " ".join(items[1:])
            sentences.append((utt_id, sentence))

    for utt_id, sentence in sentences[:1]:
        # 需要自己构造伪 batch 输入给模型
        batch = get_batch(sentence, phonemizer)
        print("batch:", batch)
        # 遍历 utt_id
        st = time.time()
        # with torch.no_grad():
        # prompt 是啥东西？？？？？？？
        # 端到端合成的时候该咋输入？

        # pred_semantic = t2s_model.model.infer(
        #     batch['phoneme_ids'].cuda(),
        #     batch['phoneme_ids_len'].cuda(),
        #     prompt.cuda(),
        #     top_k=config['inference']['top_k'])
        # print(f'{time.time() - st} sec used in T2S')

        print(f'{time.time() - st} sec used in T2S')
        np.save(output_dir / f'semantic_toks_{utt_id}.npy',
                pred_semantic.detach().cpu().numpy())


if __name__ == "__main__":
    main()
