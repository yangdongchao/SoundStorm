"""Command-line for audio compression."""
import logging
from typing import Union

import librosa
import torch
from soundstorm.s2.models.hubert.abs_tokenizer import AbsTokenizer
from soundstorm.s2.models.hubert.hubert_kmeans import HubertWithKmeans


class SemanticTokenizer(AbsTokenizer):
    def __init__(
            self,
            hubert_path,
            quantizer_path,
            duplicate: bool=False,
            device: Union[str, torch.device]=None, ):
        """  Hubert model for extract semantic token
        """
        super(SemanticTokenizer, self).__init__()
        if device is None:
            device = torch.device("cuda"
                                  if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.hubert_path = hubert_path
        self.quantizer_path = quantizer_path
        self.hubert_kmeans = HubertWithKmeans(
            checkpoint_path=self.hubert_path, kmeans_path=self.quantizer_path)
        self.hubert_kmeans = self.hubert_kmeans.to(self.device)

        logging.info(f"hubert semantic model works on {self.device}")

        # properties
        self.sr = 16 * 1000
        self.dim_codebook = 500
        self.duplicate = duplicate

    def encode(self, wav_path, sr=16000):
        # wav, sr = torchaudio.load(wav_path)
        # if sr != self.sr:
        #     wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        # 换成 librosa.load 可以直接输出经过 resample 的单通道 wav
        # wav.shape: [C, T]
        wav, sr = librosa.load(wav_path, sr=self.sr)
        assert sr == self.sr
        wav = torch.tensor(wav).unsqueeze(0)
        print("wav.shape:", wav.shape)

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
            return self.encode(wav, sr=16000)
        elif isinstance(wav, torch.Tensor):
            """
            wav.shape: [C, T]
            """
            if wav.dim() == 1:  # already done offline
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
        unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim=0)]
        return unique_arr[0]

    @property
    def codebook_length(self):
        return self.dim_codebook

    def find_length(
            self, x
    ):  # we first calculate the codec for wave x, then we get the length
        return self.tokenize(x).shape[0]


if __name__ == '__main__':
    # mHuBERT sr = 16k, hop_size = 320
    # mHuBERT 和 HuBERT 都是 16k 320
    hubert_path = './hubert_base_ls960.pt'
    quantizer_path = './hubert_base_ls960_L9_km500.bin'
    # [1, 142778]
    # LibriTTS-R 中的两条提取结果和 https://github.com/yangdongchao/SoundStorm/blob/master/data_sample/LibriTTS_1000/semantic/test.tsv
    # 中提供的在长度和数值上有些微的差别
    wav = './LJ050-0278.wav'
    wav, _ = librosa.load(wav, sr=16000)
    wav = torch.tensor(wav).unsqueeze(0)
    tokenizer = SemanticTokenizer(
        hubert_path=hubert_path, quantizer_path=quantizer_path, duplicate=True)
    # wav should be wav_path or torch.Tensor
    flat_codec = tokenizer.tokenize(wav)
    print('flat_codec:', flat_codec)
    # [445]
    # 142778/445 = 320.8494382022472
    # 142778/320 = 446.18125
    # 需要处理不整除的情况
    print('flat_codec.shape:', flat_codec.shape)
