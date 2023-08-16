# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import argparse
import logging
import os
import re
from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from soundstorm.s1.AR.data._librilight_6k import Text2SemanticDataModule
from soundstorm.s1.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from soundstorm.utils.io import load_yaml_config
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
torch.set_float32_matmul_precision('high')


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(args.config_file)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = ModelCheckpoint(
        save_top_k=-1,
        save_on_train_epoch_end=False,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir)
    logger = WandbLogger(
        project="AR_S1",
        name=output_dir.stem,
        save_dir=output_dir,
        # resume the loss curve
        resume=True,
        # id='k19kvsq8'
    )
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator='gpu',
        devices=-1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=config["train"]["precision"],
        logger=logger,
        callbacks=[ckpt_callback])

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_dirs=args.train_semantic_dirs,
        train_phoneme_dirs=args.train_phoneme_dirs,
        dev_semantic_dirs=args.dev_semantic_dirs,
        dev_phoneme_dirs=args.dev_phoneme_dirs)

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        sorted_files = sorted(
            os.listdir(ckpt_dir),
            key=lambda x: int(re.findall(r'epoch=(\d+)', x)[0]))
        ckpt_path = ckpt_dir / sorted_files[-1]
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


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
        '--train_semantic_dirs',
        type=list,
        nargs='+',
        default=["dump/small/train/"],
        help='dirs of train semantic')
    parser.add_argument(
        '--train_phoneme_dirs',
        type=list,
        nargs='+',
        default=["dump/small/train/"],
        help='dirs of train phoneme')
    parser.add_argument(
        '--dev_semantic_dirs',
        type=list,
        nargs='+',
        default=["dump/small/dev/"],
        help='dirs of dev semantic')
    parser.add_argument(
        '--dev_phoneme_dirs',
        type=list,
        nargs='+',
        default=["dump/small/dev/"],
        help='dirs of dev phoneme')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='exp/default',
        help='directory to save the results')

    args = parser.parse_args()

    new_train_semantic_dirs = []
    new_train_phoneme_dirs = []
    new_dev_semantic_dirs = []
    new_dev_phoneme_dirs = []
    # format dataset dirs
    for item in args.train_semantic_dirs:
        new_train_semantic_dirs.append(''.join(item))
    args.train_semantic_dirs = new_train_semantic_dirs

    for item in args.train_phoneme_dirs:
        new_train_phoneme_dirs.append(''.join(item))
    args.train_phoneme_dirs = new_train_phoneme_dirs

    for item in args.dev_semantic_dirs:
        new_dev_semantic_dirs.append(''.join(item))
    args.dev_semantic_dirs = new_dev_semantic_dirs

    for item in args.dev_phoneme_dirs:
        new_dev_phoneme_dirs.append(''.join(item))
    args.dev_phoneme_dirs = new_dev_phoneme_dirs

    logging.info(str(args))
    main(args)
