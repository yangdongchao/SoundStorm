# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/t2s_dataset.py
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from soundstorm.s1.AR.data.bucket_sampler import DistributedBucketSampler
from soundstorm.s1.AR.text_processing.phonemizer import GruutPhonemizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(self,
                 metadata_path: Union[str, List[str]],
                 semantic_token_path: Union[str, List[str]]) -> None:
        super().__init__()
        
        # 如何保证两种文件的 idx 一致？
        self.dataframe: pd.DataFrame = pd.read_csv(
            os.path.join(metadata_path, 'metadata.csv'), sep='|')
        self.semantic_tokens: List[List[int]] = self._read_semantic_tokens(
            semantic_token_path)
       

        assert len(self.dataframe) == len(self.semantic_tokens)

        self.phonemizer: GruutPhonemizer = GruutPhonemizer(language='en-us')
        self.max_phoneme_len: int = 256
        # pad for semantic tokens
        self.PAD: int = 1024
        self.hz=50
    
    def init_batch(self):
        return None

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict:
        row = self.dataframe.iloc[idx]

        # text (phonemes) input
        phonemes = row['Phonemes']
        # 这里对 phoneme_ids 进行截断了，如何保证和 semantic_ids 对齐？
        phoneme_ids = self.phonemizer.transform(phonemes)[:self.max_phoneme_len]
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long)
        phoneme_len = len(phoneme_ids)

        # semantic tokens target
        semantic_ids = self.semantic_tokens[idx]
        semantic_ids = torch.tensor(semantic_ids, dtype=torch.long)
        semantic_ids_len = len(semantic_ids)

        return {
            'id': idx,
            'phoneme_ids': phoneme_ids,
            'phoneme_ids_len': phoneme_len,
            'semantic_ids': semantic_ids,
            'semantic_ids_len': semantic_ids_len
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_tokens[idx]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, features: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.LongTensor] = []
        semantic_ids_lens: List[int] = []

        for feature in features:
            sample_index.append(feature["id"])
            phoneme_ids.append(feature["phoneme_ids"])
            phoneme_ids_lens.append(feature["phoneme_ids_len"])
            semantic_ids.append(feature["semantic_ids"])
            semantic_ids_lens.append(feature["semantic_ids_len"])

        batch_size: int = len(sample_index)
        max_phoneme_ids_len: int = max(phoneme_ids_lens)
        max_semantic_ids_len: int = max(semantic_ids_lens)

        # 按照 batch 里面最长的 pad 0 的过程，可以用一个函数实现
        # collate phonemes
        phoneme_ids_t: torch.Tensor = torch.zeros(
            (batch_size, max_phoneme_ids_len), dtype=torch.long)
        for i, phoneme_id_seq in enumerate(phoneme_ids):
            phoneme_ids_t[i, :phoneme_id_seq.size(0)] = phoneme_id_seq
        phoneme_ids_lens_t: torch.Tensor = torch.tensor(
            phoneme_ids_lens, dtype=torch.long)
    
        # pad 0 的过程
        # collate audio frames
        semantic_ids_t: torch.Tensor = torch.zeros(
            (batch_size, max_semantic_ids_len), dtype=torch.long) + self.PAD
        for i, frame_seq in enumerate(semantic_ids):
            semantic_ids_t[i, :frame_seq.size(0)] = frame_seq
        semantic_ids_lens_t: torch.Tensor = torch.tensor(
            semantic_ids_lens, dtype=torch.long)

        return {
            "ids": sample_index,  # List[int]
            "phoneme_ids": phoneme_ids_t,  # bs * max_phoneme_length
            "phoneme_ids_len": phoneme_ids_lens_t,  # bs
            "semantic_ids": semantic_ids_t,  # bs * max_semantic_ids_length
            "semantic_ids_len": semantic_ids_lens_t,  # bs
        }

    def _read_semantic_tokens(self,
                              semantic_token_path: str) -> List[List[int]]:
        semantic_tokens: List[List[int]] = []
        with open(semantic_token_path, 'r') as f:
            for line in f:
                semantic_tokens.append([int(x) for x in line.split(' ')])
        return semantic_tokens


if __name__ == '__main__':
    dataset = Text2SemanticDataset(
        metadata_path=[
            'data/libritts_train_clean_100', 'data/libritts_train_clean_360',
            'data/libritts_train_other_500'
        ],
        semantic_token_path=[
            'data/libritts_train_clean_100/km_semtok/libritts_train_clean_100.km',
            'data/libritts_train_clean_360/km_semtok/libritts_train_clean_360.km',
            'data/libritts_train_other_500/km_semtok/libritts_train_other_500.km'
        ])
    batch_size = 12
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        shuffle=False)
    print(dataset.__len__())
    for batch in tqdm(dataloader):
        pass
