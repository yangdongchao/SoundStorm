#

"""Command-line for audio compression."""
import argparse
from pathlib import Path
import sys
import torchaudio
import os
import torch
import typing as tp
import torch.distributed as dist
from collections import OrderedDict
from omegaconf import OmegaConf
import logging

from hubert_kmeans import HubertWithKmeans
from abs_tokenizer import AbsTokenizer

class SemanticTokenizer(AbsTokenizer):
    def __init__(self, local_rank=-1, duplicate=False):
        """  Hubert model for extract semantic token
        """
        super(SemanticTokenizer, self).__init__()
        # GPU is only for offline tokenization
        # So, when distributed training is launched, this should still be on CPU
        if not dist.is_initialized() and torch.cuda.is_available():
            local_rank -= 1 # run.pl provides rank from 1
            logging.info(f"Local rank parsed from environment variable: {local_rank}")

        if local_rank >= 0:
            self.device = torch.device(f"cuda:{local_rank}") 
        else:
            self.device = torch.device('cpu')

        self.hubert_path = './mhubert_base_vp_en_es_fr_it3.pt'
        self.quantizer_path = './mhubert_base_vp_en_es_fr_it3_L11_km1000.bin'
        self.hubert_kmeans = HubertWithKmeans(checkpoint_path=self.hubert_path, kmeans_path=self.quantizer_path)
        self.hubert_kmeans = self.hubert_kmeans.to(self.device)

        logging.info(f"hubert semantic model works on {self.device}")

        # properties
        self.sr = 16 * 1000
        self.dim_codebook = 1000
        self.duplicate = duplicate
    
    def encode(self, wav_root, sr=16000):
        wav, sr = torchaudio.load(wav_root)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = wav.to(self.device)
        flat_codec = self.hubert_kmeans(wav)
        if not self.duplicate:
            flat_codec = self.batch_unique_consecutive(flat_codec)
        flat_codec = flat_codec.to(torch.int16)
        return flat_codec

    @property
    def is_discrete(self):
        return True

    def tokenize(self, wav):
        if isinstance(wav, str):
            # if x is the wave path
            return self.encode(wav)
        elif isinstance(wav, torch.Tensor):
            if wav.dim() == 1: # already done offline
                return wav
            wav = wav.to(self.device)
            flat_codec = self.hubert_kmeans(wav)
            if not self.duplicate:
                flat_codec = self.batch_unique_consecutive(flat_codec)
            flat_codec = flat_codec.to(torch.int16)
            return flat_codec
        else:
            raise NotImplementedError

    def batch_unique_consecutive(self, t):
        t = t.unsqueeze(0)
        unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
        return unique_arr[0]
    @property
    def codebook_length(self):
        return self.dim_codebook

    def find_length(self, x): # we first calculate the codec for wave x, then we get the length
        return self.tokenize(x).shape[0]


if __name__ == '__main__':
    tokenizer = SemanticTokenizer(duplicate=True) 
    # wav = tokenizer.decode(codec)
    # print(wav.shape)
