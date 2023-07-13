# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import argparse
import logging
from typing import Dict

import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from soundstorm.s1.AR.data.data_module import Text2SemanticDataModule
from soundstorm.s1.AR.models.t2s_lightning_module import Text2SemanticLightningModule
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
torch.set_float32_matmul_precision('high')


def main(args):

    with open(args.path_to_config, "r") as f:
        configuration: Dict = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(configuration["train"]["seed"], workers=True)
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        save_top_k=-1,
        save_on_train_epoch_end=False,
        every_n_epochs=configuration["train"]["save_every_n_epoch"])
    trainer: Trainer = Trainer(
        max_epochs=configuration["train"]["epochs"],
        accelerator='gpu',
        devices=-1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=configuration["train"]["precision"],
        logger=WandbLogger(project="soundstorm"),
        callbacks=[checkpoint_callback],
        use_distributed_sampler=False)
    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        configuration)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        configuration)
    trainer.fit(model, data_module)


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_config", type=str, default='configuration/ljspeech_t2s.yaml')
    args = parser.parse_args()
    logging.info(str(args))
    main(args)
