# ------------------------------------------
# Diffsound
# code based on https://github.com/cientgu/VQ-Diffusion
# ------------------------------------------
import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from soundstorm.s2.utils.misc import instantiate_from_config
from torch import nn
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 2)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=(1, 2),
                 bilinear=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = torch.nn.functional.interpolate(
            x1, scale_factor=self.scale_factor, mode="nearest")
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FullAttention(nn.Module):
    def __init__(
            self,
            n_embd,  # the embed dim
            n_head,  # the number of heads
            attn_pdrop=0.1,  # attention dropout prob
            resid_pdrop=0.1,  # residual attention dropout prob
            causal=True, ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal

    def forward(self, x, encoder_output, mask=None):
        if mask is not None:
            slf_mask = mask.unsqueeze(1).repeat(1, self.n_head, 1,
                                                1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        B, T, C = x.size()
        # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1,
                                                                            2)
        # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1,
                                                                              2)
        # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1,
                                                                              2)
        # (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 对 attention 权重进行 mask
        # add mask
        if mask is not None:
            att = att.masked_fill(slf_mask, -np.inf)
        # (B, nh, T, T)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        # re-assemble all head outputs side by side, (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # (B, T, T)
        att = att.mean(dim=1, keepdim=False)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(
            self,
            # the embed dim
            n_embd,
            # condition dim
            condition_embd,
            # the number of heads 
            n_head,
            # attention dropout prob
            attn_pdrop=0.1,
            # residual attention dropout prob
            resid_pdrop=0.1, ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        # causal mask to ensure that attention is only applied to the left in the input sequence

    def forward(self, x, encoder_output, mask=None):
        # print('encoder_output ', encoder_output.shape)
        # print('c_mask ', c_mask.shape)
        if mask is not None:
            # (n*b) x .. x ..
            slf_mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        else:
            slf_mask = None
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (B, nh, T_E, hs)
        k = self.key(encoder_output).view(B, T_E, self.n_head,
                                          C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1,
                                                                              2)
        # (B, nh, T_E, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head,
                                            C // self.n_head).transpose(1, 2)
        # (B, nh, T, T_E)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # print('att ', att.shape)
        # print('v ', v.shape)
        # print('slf_mask ', slf_mask.shape)
        # print('smask ', slf_x_mask.shape)
        # add mask
        if slf_mask is not None:
            # 对 attention 权重进行 mask
            att = att.masked_fill(slf_mask, -np.inf)
        # (B, nh, T, T)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        # re-assemble all head outputs side by side, (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # (B, T, T)
        att = att.mean(dim=1, keepdim=False)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adainsnorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (
            1 + scale) + shift
        return x


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(
            self,
            class_type='adalayernorm',
            class_number=1000,
            n_embd=1024,
            n_head=16,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            activate='GELU',
            attn_type='full',
            if_upsample=False,
            condition_dim=1024,
            diffusion_step=100,
            timestep_type='adalayernorm',
            mlp_type='fc', ):
        super().__init__()
        self.if_upsample = if_upsample
        self.attn_type = attn_type
        if attn_type in ['selfcross', 'selfcondition', 'self']:
            if 'adalayernorm' in timestep_type:
                self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
        else:
            self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.if_selfcross = False
        if attn_type in ['self', 'selfcondition']:
            self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop, )
        elif attn_type == 'selfcross':
            self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop, )
            self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop, )
            if 'adalayernorm' in timestep_type:
                self.ln1_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
        else:
            print("attn_type error")
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'conv_mlp':
            self.mlp = Conv_MLP(n_embd, mlp_hidden_times, act, resid_pdrop)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop), )

    def forward(self, x, encoder_output, x_mask, cond_emb_mask, timestep):
        # encoder_output denotes the conditional information
        # get the max len
        max_len = x.shape[1]
        # get self attention mask 
        slf_x_attn_mask = x_mask.unsqueeze(1).expand(-1, max_len, -1)
        # using cross atten
        if self.attn_type == "selfcross":
            a, att = self.attn1(
                self.ln1(x, timestep), encoder_output, mask=slf_x_attn_mask)
            # print('a ', a.shape)
            if x_mask is not None:
                a = a.masked_fill(x_mask.unsqueeze(-1), 0)
                # print('x_mask.unsqueeze(-1) ', x_mask.unsqueeze(-1).shape)
            x = x + a
            # slf_x_attn_mask2 = x_mask.unsqueeze(2).expand(-1, -1, encoder_output.shape[1])
            # x_mask=slf_x_attn_mask2, c_mask=cond_emb_mask
            a, att = self.attn2(
                self.ln1_1(x, timestep), encoder_output, mask=None)
            if x_mask is not None:
                a = a.masked_fill(x_mask.unsqueeze(-1), 0)
            x = x + a
        elif self.attn_type == "selfcondition":
            a, att = self.attn(
                self.ln1(x, timestep), encoder_output, mask=slf_x_attn_mask)
            if x_mask is not None:
                a = a.masked_fill(x_mask.unsqueeze(-1), 0)
            x = x + a
            # only one really use encoder_output
            x = x + self.mlp(x + encoder_output)
            return x, att
        # 'self'
        else:
            a, att = self.attn(
                self.ln1(x, timestep), encoder_output, mask=slf_x_attn_mask)
            if x_mask is not None:
                a = a.masked_fill(x_mask.unsqueeze(-1), 0)
            x = x + a
        x = x + self.mlp(self.ln2(x))
        # ? 还需要嘛？
        if x_mask is not None:
            x = x.masked_fill(x_mask.unsqueeze(-1), 0)
        return x, att


class Conv_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_embd,
            out_channels=int(mlp_hidden_times * n_embd),
            kernel_size=3,
            stride=1,
            padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(
            in_channels=int(mlp_hidden_times * n_embd),
            out_channels=n_embd,
            kernel_size=3,
            stride=1,
            padding=1)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        n = x.size()[1]
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(n)))
        x = self.conv2(self.act(self.conv1(x)))
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.dropout(x)


class Classifer(nn.Module):
    def __init__(self, n_embd, out_cls):
        super().__init__()
        self.LN = nn.LayerNorm(n_embd)
        self.Linear = nn.Linear(n_embd, out_cls)

    def forward(self, x):
        x = self.LN(x)
        return self.Linear(x)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class Text2ImageTransformer(nn.Module):
    def __init__(
            self,
            n_layer=14,
            n_q=2,
            n_embd=1024,
            n_head=16,
            attn_pdrop=0,
            resid_pdrop=0,
            mlp_hidden_times=4,
            block_activate=None,
            attn_type='selfcross',
            condition_dim=512,
            diffusion_step=1000,
            timestep_type='adalayernorm',
            content_emb_config=None,
            condition_emb_config=None,
            mlp_type='fc',
            checkpoint=False, ):
        super().__init__()
        self.use_checkpoint = checkpoint
        self.n_q = n_q
        # when init the model, the number of q has add 1
        self.content_emb = instantiate_from_config(content_emb_config)
        # self.condition_emb = instantiate_from_config(condition_emb_config) 
        # 用于semantic token
        self.semantic_embedding = nn.Embedding(
            1000 + 4, content_emb_config['params']['embed_dim'])
        # transformer
        #assert attn_type == 'selfcross'
        self.inc = (DoubleConv(condition_dim, condition_dim))
        self.down1 = (Down(condition_dim, condition_dim, kernel_size=(3, 1)))
        # self.down2 = (Down(condition_dim, condition_dim, kernel_size=(2,1)))
        # self.down3 = (Down(condition_dim, condition_dim, kernel_size=(2,1)))

        self.up1 = (Up(condition_dim * 2, condition_dim, scale_factor=(3, 1)))
        # self.up2 = (Up(condition_dim*2, condition_dim, scale_factor=(2,1)))
        # self.up3 = (Up(condition_dim*2, condition_dim, scale_factor=(2,1)))

        all_attn_type = [attn_type] * n_layer
        self.blocks = nn.Sequential(* [
            Block(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                attn_type=all_attn_type[n],
                condition_dim=condition_dim,
                diffusion_step=diffusion_step,
                timestep_type=timestep_type,
                mlp_type=mlp_type, ) for n in range(n_layer)
        ])
        # final prediction head
        self.semantic_token_nums = 1000
        self.prompt_semantic_start_id = self.semantic_token_nums
        self.prompt_semantic_end_id = self.semantic_token_nums + 1
        self.target_semantic_start_id = self.semantic_token_nums + 2
        self.target_semantic_end_id = self.semantic_token_nums + 3

        self.acoustic_token_nums = 1024  # 
        # self.prompt_acoustic_start_id = self.acoustic_token_nums 
        # self.prompt_acoustic_end_id = self.acoustic_token_nums + 1
        # self.target_acoustic_start_id = self.acoustic_token_nums + 2
        # self.target_acoustic_end_id = self.acoustic_token_nums + 3
        # 最长的序列假设为10s
        self.prompt_semantic_pos_emb = LearnedPositionEmbeddings(500, n_embd)
        # 20s
        self.target_semantic_pos_emb = LearnedPositionEmbeddings(1000, n_embd)
        self.prompt_acoustic_pos_emb = LearnedPositionEmbeddings(500, n_embd)
        # 20s
        self.target_acoustic_pos_emb = LearnedPositionEmbeddings(3000, n_embd)
        # num_embed: 2887
        out_cls = self.content_emb.num_embed - 1
        self.register_buffer('cls_ids', torch.arange(out_cls))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine is True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(
                            m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(
                            m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = [
                'pos_emb', 'width_emb', 'height_emb', 'pad_emb',
                'token_type_emb'
            ]
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(
                                    getattr(getattr(self, mn), pn),
                                    torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {
                pn: p
                for pn, p in self.transformer.named_parameters()
            }  # if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(
                inter_params
            ) == 0, "parameters %s made it into both decay/no_decay sets!" % (
                str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )
            # create the pytorch optimizer object
            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(decay))],
                    "weight_decay": 0.01
                },
                {
                    "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                    "weight_decay": 0.0
                },
            ]
            return optim_groups

    def forward(self, input, condition, x_mask, cond_emb_mask, t):
        # cont_emb: B, n_q, len, dim
        cont_emb, pos_emb = self.content_emb(input)
        cont_emb = cont_emb.permute(0, 3, 1, 2)
        x1 = self.inc(cont_emb)
        x2 = self.down1(x1)

        emb = x2.transpose(1, 3)
        # B, len , dim
        emb = emb.reshape(emb.shape[0], -1, emb.shape[3])
        emb = emb + pos_emb
        #emb = cont_emb
        prompt_semantic_token_ids = condition['prompt_semantics']
        target_semantic_token_ids = condition['target_semantics']
        prompt_acoustics = condition['prompt_acoustics']
        # only use the first codebook for speaker 
        prompt_acoustic_token_ids = prompt_acoustics[:, 0, :]
        prompt_semantic_token_ids = rearrange(prompt_semantic_token_ids,
                                              'b ... -> b (...)')
        # transfer to [B, T]
        target_semantic_token_ids = rearrange(target_semantic_token_ids,
                                              'b ... -> b (...)')
        # print('prompt_semantic_token_ids ', prompt_semantic_token_ids.shape)
        # print('target_semantic_token_ids ', target_semantic_token_ids.shape)
        # print('prompt_acoustic_token_ids ', prompt_acoustic_token_ids.shape)
        # 增加和一个stop token
        prompt_semantic_token_ids = F.pad(
            prompt_semantic_token_ids, (0, 1),
            value=self.prompt_semantic_end_id)
        target_semantic_token_ids = F.pad(
            target_semantic_token_ids, (0, 1),
            value=self.target_semantic_end_id)

        prompt_semantic_token_ids = F.pad(
            prompt_semantic_token_ids, (1, 0),
            value=self.prompt_semantic_start_id)
        target_semantic_token_ids = F.pad(
            target_semantic_token_ids, (1, 0),
            value=self.target_semantic_start_id)

        prompt_semantic_token_emb = self.semantic_embedding(
            prompt_semantic_token_ids.long())
        target_semantic_token_emb = self.semantic_embedding(
            target_semantic_token_ids.long())
        # promp
        prompt_acoustic_token_emb = self.content_emb.embs['0'](
            prompt_acoustic_token_ids.long())

        prompt_semantic_token_emb = prompt_semantic_token_emb + self.prompt_semantic_pos_emb(
            prompt_semantic_token_emb)
        target_semantic_token_emb = target_semantic_token_emb + self.target_semantic_pos_emb(
            target_semantic_token_emb)
        prompt_acoustic_token_emb = prompt_acoustic_token_emb + self.prompt_acoustic_pos_emb(
            prompt_acoustic_token_emb)
        # print('prompt_semantic_token_emb ', prompt_semantic_token_emb.shape)
        # print('target_semantic_token_emb ', target_semantic_token_emb.shape)
        # print('prompt_acoustic_token_emb ', prompt_acoustic_token_emb.shape)
        # [B, all, 512]      
        cond_emb = torch.cat(
            (prompt_semantic_token_emb, target_semantic_token_emb,
             prompt_acoustic_token_emb),
            dim=1)

        for block_idx in range(len(self.blocks)):
            if self.use_checkpoint is False:
                # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
                emb, att_weight = self.blocks[block_idx](
                    emb, cond_emb, x_mask, cond_emb_mask, t.cuda())
            else:
                emb, att_weight = checkpoint(self.blocks[block_idx], emb,
                                             cond_emb, x_mask, cond_emb_mask,
                                             t.cuda())
        x3 = emb.unsqueeze(1).permute(0, 3, 1, 2)
        x = self.up1(x3, x1)  # 512
        x = x.permute(0, 2, 3, 1)

        logits = []
        index_mat = self.cls_ids
        # 1, N
        index_mat = index_mat.unsqueeze(0).repeat(x.shape[0], 1)
        for index in range(self.n_q):
            weight = self.content_emb.embs[str(index)](index_mat)
            tmp_logit = torch.bmm(x[:, index, :, :], weight.transpose(1, 2))
            logits.append(tmp_logit)
        logits = torch.cat(logits, dim=1)
        out = rearrange(logits, 'b l c -> b c l')
        return out
