# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/model/t2s_lightning_module.py
from typing import Dict

import pytorch_lightning
import torch
from soundstorm.s1.AR.models.t2s_model import Text2SemanticDecoder
from soundstorm.s1.AR.modules.lr_schedulers import WarmupCosineLRSchedule
from soundstorm.s1.AR.modules.optim import ScaledAdam


class Text2SemanticLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Text2SemanticDecoder(config=config)
        self.automatic_optimization = False
        self.save_hyperparameters()

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
            "acc_t2s_top10", acc, on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Dict, batch_idx: int):

        semantic_len = batch['semantic_ids'].size(1)
        prompt_len = min(int(semantic_len * 0.5), 150)
        prompt = batch['semantic_ids'][:, :prompt_len]
        pred_semantic = self.model.infer(batch['phoneme_ids'],
                                         batch['phoneme_ids_len'], prompt)
        torch.save(pred_semantic.detach().cpu(),
                   f'eval/semantic_toks_{batch_idx}.pt')
        if batch_idx == 0:
            print('')

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
