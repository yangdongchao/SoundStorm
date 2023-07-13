# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/data_module.py
from pytorch_lightning import LightningDataModule
from soundstorm.s1.AR.data.bucket_sampler import DistributedBucketSampler
from soundstorm.s1.AR.data.dataset import Text2SemanticDataset
from torch.utils.data import DataLoader


class Text2SemanticDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
        self._train_dataset = Text2SemanticDataset(
            metadata_path=self.config['data']['train_metadata_path'],
            semantic_token_path=self.config['data'][
                'train_semantic_token_path'])
        self._test_dataset = Text2SemanticDataset(
            metadata_path=self.config['data']['eval_metadata_path'],
            semantic_token_path=self.config['data']['eval_semantic_token_path'],
            max_sample=self.config['data']['max_eval_sample'])

    def train_dataloader(self):
        batch_size = self.config['train']['batch_size']
        sampler = DistributedBucketSampler(
            self._train_dataset, batch_size=batch_size)
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._train_dataset.collate)

    def val_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate)

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate)
