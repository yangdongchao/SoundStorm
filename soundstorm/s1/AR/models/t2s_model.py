# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/model/t2s_model.py
import torch
from soundstorm.s1.AR.models.utils import make_pad_mask
from soundstorm.s1.AR.models.utils import topk_sampling
from soundstorm.s1.AR.modules.embedding import SinePositionalEmbedding
from soundstorm.s1.AR.modules.embedding import TokenEmbedding
from soundstorm.s1.AR.modules.transformer import LayerNorm
from soundstorm.s1.AR.modules.transformer import TransformerEncoder
from soundstorm.s1.AR.modules.transformer import TransformerEncoderLayer
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024
}


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config['model']["hidden_dim"]
        self.embedding_dim = config['model']["embedding_dim"]
        self.num_head = config['model']["head"]
        self.num_layers = config['model']["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config['model']["vocab_size"]
        self.phoneme_vocab_size = config['model']["phoneme_vocab_size"]
        self.p_dropout = config['model']["dropout"]
        self.EOS = config['model']["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        assert self.EOS == 1024

        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True)

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first, ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None, )

        self.ar_predict_layer = nn.Linear(
            self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS, )

    def forward(self, x, x_lens, y, y_lens):

        x = self.ar_text_embedding(x)
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True, )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1, ),
            (x_len, 0),
            value=False, )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (ar_xy_padding_mask.view(bsz, 1, 1, src_len)
                            .expand(-1, self.num_head, -1, -1)
                            .reshape(bsz * self.num_head, 1, src_len))
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask

        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask, )
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
        # loss
        loss = F.cross_entropy(logits, targets, reduction='sum')
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc

    def infer(self, x, x_lens, prompts, top_k=-100):

        x = self.ar_text_embedding(x)
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        while True:

            y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)

            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True, )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                (x_len, 0),
                value=False, )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0).to(y.device)

            xy_dec, _ = self.h(
                (xy_pos, None),
                mask=xy_attn_mask, )
            logits = self.ar_predict_layer(xy_dec[:, -1])

            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=1.0)

            if (torch.argmax(logits, dim=-1)[0] == self.EOS or
                    samples[0, 0] == self.EOS or
                (y.shape[1] - prefix_len) > 512):
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print('bad zero prediction')

                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            y = torch.concat([y, samples], dim=1)
        return y

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(
            y, (0, 1), value=0) + eos_id * F.pad(
                y_mask_int, (0, 1), value=1)

        return targets[:, :-1], targets[:, 1:]
