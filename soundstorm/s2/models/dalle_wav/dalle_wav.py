# ------------------------------------------
# Diffsound
# written by Dongchao Yang
# ------------------------------------------
import torch
from soundstorm.s2.utils.misc import instantiate_from_config
from soundstorm.utils.initialize import initialize
from torch import nn


class DALLE(nn.Module):
    def __init__(self,
                 diffusion_config,
                 n_q: int=4,
                 init_type: str="kaiming_uniform"):
        super().__init__()
        self.n_q = n_q
        # we donot use the classifier guidance in this stage
        self.guidance_scale = 1.0
        # self.content_codec = instantiate_from_config(content_codec_config)
        diffusion_config['params']['n_q'] = self.n_q
        self.transformer = instantiate_from_config(diffusion_config)
        self.truncation_forward = False
        initialize(self, init_type)

    def parameters(self, recurse: bool=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                # the parameters() method is not overwritten for some classes
                try:
                    params += getattr(self, name).parameters(
                        recurse=recurse, name=name)
                except Exception:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    @property
    def device(self):
        return self.transformer.device

    def get_ema_model(self):
        return self.transformer

    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))

        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out = func(out, args[1], args[2], **kwards)
            return out

        return wrapper

    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            content_codec = self.content_codec
            save_path = self.this_save_path

            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k=truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs

            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))

            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True)
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:, 0:1, :], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:, :-1, :]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float() * out + (1 - temp4.float()) * (-70)
                probs = temp5
                return probs

            return wrapper
        else:
            print("wrong sample type")

    @torch.no_grad()
    def generate_content(self,
                         batch,
                         condition=None,
                         filter_ratio: float=0.0,
                         replicate: int=1,
                         sample_type: str="top0.85r"):
        self.eval()
        con = batch['target_acoustics']
        batch_size = con.shape[0]

        def cf_predict_start(log_x_t, cond_emb, x_mask, cond_emb_mask, t):
            log_x_recon = self.transformer.predict_start(
                log_x_t, cond_emb, x_mask, cond_emb_mask, t)[:, :-1]
            if abs(self.guidance_scale - 1) < 1e-3:
                zero_vector = torch.zeros(
                    x_mask.shape[0], 1,
                    log_x_recon.shape[2]).type_as(log_x_t) - 70
                return torch.cat((log_x_recon, zero_vector), dim=1)
            cf_cond_emb = self.transformer.empty_text_embed[:cond_emb.shape[
                1], :].unsqueeze(0).repeat(batch_size, 1, 1)
            cf_log_x_recon = self.transformer.predict_start(
                log_x_t, cf_cond_emb.type_as(cond_emb), x_mask, cond_emb_mask,
                t)[:, :-1]
            log_new_x_recon = cf_log_x_recon + self.guidance_scale * (
                log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(
                log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            zero_vector = torch.zeros(
                x_mask.shape[0], 1,
                log_new_x_recon.shape[2]).type_as(log_x_t) - 70
            log_pred = torch.cat((log_new_x_recon, zero_vector), dim=1)
            return log_pred

        out = {}
        # 重复多少次?
        if replicate != 1:
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat(
                        [condition[k] for _ in range(replicate)], dim=0)
        content_token = None
        # using r,fast
        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1] == 'q':
                self.transformer.p_sample = self.p_sample_with_truncation(
                    self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[
                0][:3] == "top" and self.truncation_forward is False:
            self.transformer.cf_predict_start = self.predict_start_with_truncation(
                cf_predict_start, sample_type.split(',')[0])
            self.truncation_forward = True
        if len(sample_type.split(',')) == 2 and sample_type.split(',')[
                1][:4] == 'fast':
            trans_out = self.transformer.sample_fast(
                batch=batch,
                filter_ratio=filter_ratio,
                return_logits=False,
                skip_step=int(sample_type.split(',')[1][4:]))
        else:
            trans_out = self.transformer.sample(
                batch=batch, filter_ratio=filter_ratio, return_logits=False)
        out['token_pred'] = trans_out['pre_content_token']
        return out

    @torch.no_grad()
    def infer_one(self, batch, sample_type: str="top0.85r"):
        # sample_type = "top0.85r,fast1" for fast inference
        output = self.generate_content(batch, sample_type=sample_type)
        return output

    def forward(self, batch, **kwargs):
        # 信息处理直接交给 transformer
        output = self.transformer(batch, **kwargs)
        return output
