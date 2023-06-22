import torch
import torch.nn as nn


class BaseEmbedding(nn.Module):
    def get_loss(self):
        return None

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            super().train()
        return self

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()


class DalleMaskImageEmbedding(BaseEmbedding):
    def __init__(
            self,
            # should be quantize_number
            num_embed=1024,
            # height and with 
            max_size=1024,
            embed_dim=3968,
            n_q=4,
            trainable=True,
            pos_emb_type='embedding'):
        super().__init__()
        self.max_size = max_size
        # add a mask token
        self.num_embed = num_embed + 1
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.n_q = n_q
        self.pos_emb_type = pos_emb_type
        assert self.pos_emb_type in ['embedding', 'parameter']
        self.embs = nn.ModuleDict({
            '0': nn.Embedding(self.num_embed, embed_dim),
            '1': nn.Embedding(self.num_embed, embed_dim),
            '2': nn.Embedding(self.num_embed, embed_dim),
        })
        self.register_buffer('position_ids', torch.arange(max_size))
        if self.pos_emb_type == 'embedding':
            self.pos_emb = nn.Embedding(self.max_size, embed_dim)
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.max_size, embed_dim))
        self._set_trainable()

    def forward(self, index, **kwargs):
        # B x L
        assert index.dim() == 2
        index = index.reshape(index.shape[0], self.n_q, -1)
        index[index < 0] = 0
        emb = []
        for b in range(index.shape[0]):
            sub_emb = []
            for i in range(self.n_q):
                tmp_emb = self.embs[str(i)](index[b, i, :])
                sub_emb.append(tmp_emb.unsqueeze(0))
            # n_q, len, dim
            sub_emb = torch.cat(sub_emb, dim=0)
            emb.append(sub_emb.unsqueeze(0))
        target_emb = torch.cat(emb, dim=0)
        if target_emb.shape[2] > 0:
            if self.pos_emb_type == 'embedding':
                position_ids = self.position_ids[:target_emb.shape[2]]
            else:
                print('Not support non-embedding')
                assert 1 == 2
            # emb = emb + self.pos_emb(position_ids)[None,:,:]
        return target_emb, self.pos_emb(position_ids)[None, :, :]
