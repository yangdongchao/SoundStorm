# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import argparse
import logging
from typing import Dict
from pathlib import Path


import torch
import yaml
import wandb

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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(dir=output_dir, resume='allow')
    wandb.run.name = output_dir.stem
    wandb.finish()

    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config_file, "r") as f:
        config: Dict = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = ModelCheckpoint(
        save_top_k=-1,
        save_on_train_epoch_end=False,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir)
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator='gpu',
        devices=-1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=config["train"]["precision"],
        logger=WandbLogger(project="AR_S1"),
        callbacks=[ckpt_callback])

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=args.train_semantic_path,
        train_phoneme_path=args.train_phoneme_path,
        dev_semantic_path=args.dev_semantic_path,
        dev_phoneme_path=args.dev_phoneme_path)
    trainer.fit(model, data_module)


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/default.yaml',
        help='path of config file')
    # args for dataset
    parser.add_argument(
        '--train_semantic_path',
        type=str,
        default='dump/train/semantic_token.tsv')
    parser.add_argument(
        '--train_phoneme_path', type=str, default='dump/train/phonemes.npy')
    parser.add_argument(
        '--dev_semantic_path', type=str, default='dump/dev/semantic_token.tsv')
    parser.add_argument(
        '--dev_phoneme_path', type=str, default='dump/dev/phonemes.npy')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='exp/default',
        help='directory to save the results')
    
    args = parser.parse_args()
    logging.info(str(args))
    main(args)
