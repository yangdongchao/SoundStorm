# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/model/t2s_lightning_module.py
import os
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from soundstorm.s1.AR.models.t2s_model import Text2SemanticDecoder
from soundstorm.s1.AR.modules.lr_schedulers import WarmupCosineLRSchedule
from soundstorm.s1.AR.modules.optim import ScaledAdam


class Text2SemanticLightningModule(LightningModule):
    def __init__(self, config, output_dir):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.eval_dir = output_dir / 'eval'
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch: Dict, batch_idx: int):

        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        loss, acc = self.model.forward(
            batch['phoneme_ids'], batch['phoneme_ids_len'],
            batch['semantic_ids'], batch['semantic_ids_len'])
        self.manual_backward(loss)

        if batch_idx > 0 and batch_idx % 4 == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True)
        self.log(
            "top_" + str(self.top_k) + "_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True)

    def validation_step(self, batch: Dict, batch_idx: int):
        # get loss
        loss, acc = self.model.forward(
            batch['phoneme_ids'], batch['phoneme_ids_len'],
            batch['semantic_ids'], batch['semantic_ids_len'])
        
        self.log("val_total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val_top_" + str(self.top_k) + "_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True)

        # get infer output
        semantic_len = batch['semantic_ids'].size(1)
        prompt_len = min(int(semantic_len * 0.5), 150)
        prompt = batch['semantic_ids'][:, :prompt_len]
        pred_semantic = self.model.infer(batch['phoneme_ids'],
                                         batch['phoneme_ids_len'], prompt)
        save_name = f'semantic_toks_{batch_idx}.pt'
        save_path = os.path.join(self.eval_dir, save_name)
        torch.save(pred_semantic.detach().cpu(), save_path)

    def configure_optimizers(self):
        model_parameters = self.model.parameters()
        parameters_names = []
        parameters_names.append([
            name_param_pair[0]
            for name_param_pair in self.model.named_parameters()
        ])
        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000, )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler":
                WarmupCosineLRSchedule(
                    lm_opt,
                    init_lr=self.config['optimizer']['lr_init'],
                    peak_lr=self.config['optimizer']['lr'],
                    end_lr=self.config['optimizer']['lr_end'],
                    warmup_steps=self.config['optimizer']['warmup_steps'],
                    total_steps=self.config['optimizer']['decay_steps'])
            }
        }
