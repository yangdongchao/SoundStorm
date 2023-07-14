# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/data_module.py
from pytorch_lightning import LightningDataModule
from soundstorm.s1.AR.data.bucket_sampler import DistributedBucketSampler
from soundstorm.s1.AR.data.dataset import Text2SemanticDataset
from torch.utils.data import DataLoader


class Text2SemanticDataModule(LightningDataModule):
    def __init__(self, config, train_semantic_path, train_phoneme_path,
                 dev_semantic_path, dev_phoneme_path):
        super().__init__()
        self.config = config
        self.train_semantic_path = train_semantic_path
        self.train_phoneme_path = train_phoneme_path
        self.dev_semantic_path = dev_semantic_path
        self.dev_phoneme_path = dev_phoneme_path
        self.num_workers = self.config['data']['num_workers']

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
        self._train_dataset = Text2SemanticDataset(
            phoneme_path=self.train_phoneme_path,
            semantic_path=self.train_semantic_path)
        self._dev_dataset = Text2SemanticDataset(
            phoneme_path=self.dev_phoneme_path,
            semantic_path=self.dev_semantic_path,
            max_sample=self.config['data']['max_eval_sample'])

    def train_dataloader(self):
        batch_size = self.config['train']['batch_size']
        sampler = DistributedBucketSampler(
            self._train_dataset, batch_size=batch_size)
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._train_dataset.collate,
            num_workers=self.num_workers,)

    def val_dataloader(self):
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
            num_workers=self.num_workers,)

    # 这个会使用到嘛？
    def test_dataloader(self):
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate)
