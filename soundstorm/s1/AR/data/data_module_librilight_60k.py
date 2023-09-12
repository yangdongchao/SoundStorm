# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/data_module.py
import random
import time

from pytorch_lightning import LightningDataModule
from soundstorm.s1.AR.data.dataset_librilight_60k import DDPSyncSampler
from soundstorm.s1.AR.data.dataset_librilight_60k import Text2SemanticDataset
from torch.utils.data import DataLoader


class Text2SemanticDataModule(LightningDataModule):
    def __init__(
            self,
            config,
            train_semantic_paths,
            train_phoneme_paths,
            dev_semantic_paths,
            dev_phoneme_paths,
            train_non_speech_paths=None,
            dev_non_speech_paths=None,
            global_rank=0,
            local_rank=0,
            world_size=8, ):
        super().__init__()
        self.config = config
        self.train_semantic_paths = train_semantic_paths
        self.train_phoneme_paths = train_phoneme_paths
        self.dev_semantic_paths = dev_semantic_paths
        self.dev_phoneme_paths = dev_phoneme_paths
        self.train_non_speech_paths = train_non_speech_paths
        self.dev_non_speech_paths = dev_non_speech_paths
        self.num_workers = self.config['data']['num_workers']
        self.global_rank = global_rank
        print("self.global_rank:", self.global_rank)
        self.local_rank = local_rank
        print("self.local_rank:", self.local_rank)
        self.world_size = world_size
        print("self.world_size:", self.world_size)
        self.persistent_workers = True if self.num_workers > 0 else False
        self.prefetch_factor = 2

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
        start_build_time = time.time()
        self._train_dataset = Text2SemanticDataset(
            phoneme_paths=self.train_phoneme_paths,
            semantic_paths=self.train_semantic_paths,
            non_speech_paths=self.train_non_speech_paths,
            max_sec=self.config['data']['max_sec'],
            pad_val=self.config['data']['pad_val'],
            min_ps_ratio=self.config['data'].get('min_ps_ratio', 6),
            max_ps_ratio=self.config['data'].get('max_ps_ratio', 22), )
        self._dev_dataset = Text2SemanticDataset(
            phoneme_paths=self.dev_phoneme_paths,
            semantic_paths=self.dev_semantic_paths,
            non_speech_paths=self.dev_non_speech_paths,
            max_sample=self.config['data']['max_eval_sample'],
            max_sec=self.config['data']['max_sec'],
            pad_val=self.config['data']['pad_val'],
            min_ps_ratio=self.config['data'].get('min_ps_ratio', 6),
            max_ps_ratio=self.config['data'].get('max_ps_ratio', 22), )
        print(
            f"time of build dataloader: {round(time.time() - start_build_time, 2)}s"
        )

    def train_dataloader(self):
        batch_size = self.config['train']['batch_size']
        seed = 999
        # Make sure this is identical to each GPU
        random.seed(seed)

        train_sampler = DDPSyncSampler(
            size=self._train_dataset.__len__(),
            seed=seed,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            shuffle=True)
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=self._train_dataset.collate,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True, 
            drop_last=True, )

    def val_dataloader(self):
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True, 
            drop_last=True, )

    # 这个会使用到嘛？
    def test_dataloader(self):
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate)
