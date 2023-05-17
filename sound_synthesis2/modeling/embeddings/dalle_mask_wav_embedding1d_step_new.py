import torch
import torch.nn as nn
from .base_embedding import BaseEmbedding

class DalleMaskImageEmbedding(BaseEmbedding):
    def __init__(self,
                 num_embed=1024, # #should be quantize_number
                 max_size= 1024, # height and with 
                 embed_dim=3968, 
                 n_q = 4,
                 trainable=True,
                 pos_emb_type='embedding'
        ):
        super().__init__()
        self.max_size = max_size
        self.num_embed = num_embed + 1 # add a mask token
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.n_q = n_q
        self.pos_emb_type = pos_emb_type
        assert self.pos_emb_type in ['embedding', 'parameter']
        self.emb = nn.ModuleDict({
            '0': nn.Embedding(self.num_embed, embed_dim),
            '1': nn.Embedding(self.num_embed, embed_dim),
            '2': nn.Embedding(self.num_embed, embed_dim),
            '3': nn.Embedding(self.num_embed, embed_dim),
            '4': nn.Embedding(self.num_embed, embed_dim),
            '5': nn.Embedding(self.num_embed, embed_dim),
            '6': nn.Embedding(self.num_embed, embed_dim),
            '7': nn.Embedding(self.num_embed, embed_dim)
        }) 
        # self.spatial_emb = nn.Embedding(n_q, embed_dim)
        # self.register_buffer('spatial_ids', torch.arange(n_q))
        self.register_buffer('position_ids', torch.arange(max_size))
        if self.pos_emb_type == 'embedding':
            self.pos_emb = nn.Embedding(self.max_size, embed_dim)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_size, embed_dim))
        self._set_trainable()

    def forward(self, index, x_0, s, empty_mask, **kwargs):
        assert index.dim() == 2 # B x L
        if x_0.shape[1] == index.shape[1]: # training
            index = index.reshape(index.shape[0], self.n_q, -1) # 
            if x_0.dim()== 2: # transfer to 3 dimension
                x_0 = x_0.reshape(x_0.shape[0], self.n_q, -1)
            index[index < 0] = 0  
            x_0[x_0<0] = 0
            x_0 = x_0.long()
            emb = []
            target_x0 = []
            for b in range(index.shape[0]):
                tmp_emb = self.emb[str(s[b].item())](index[b,s[b],:])
                emb.append(tmp_emb.unsqueeze(0))
                tmp_x0 = torch.zeros_like(tmp_emb) # all zero
                if empty_mask[b]:
                    # 若是空文本，则不加任何的条件信息
                    target_x0.append(tmp_x0.unsqueeze(0))
                else:
                    for i in range(s[b]):
                        x_0_emb = self.emb[str(i)](x_0[b, i, :]) # transfer x0
                        tmp_x0 += x_0_emb
                    target_x0.append(tmp_x0.unsqueeze(0))
            target_emb = torch.cat(emb, dim=0)
            target_x0 = torch.cat(target_x0, dim=0)
        else:
            if x_0.dim()== 2: # transfer to 3 dimension
                x_0 = x_0.reshape(x_0.shape[0], self.n_q, -1)
            index[index < 0] = 0  
            x_0[x_0<0] = 0
            x_0 = x_0.long()
            emb = []
            target_x0 = []
            for b in range(index.shape[0]):
                tmp_emb = self.emb[str(s[b].item())](index[b,:])
                emb.append(tmp_emb.unsqueeze(0))
                tmp_x0 = torch.zeros_like(tmp_emb) # all zero
                if empty_mask[b]:
                    # 若是空文本，则不加任何的条件信息
                    target_x0.append(tmp_x0.unsqueeze(0))
                else:
                    for i in range(s[b]):
                        x_0_emb = self.emb[str(i)](x_0[b, i, :]) # transfer x0
                        tmp_x0 += x_0_emb
                    target_x0.append(tmp_x0.unsqueeze(0))
            target_emb = torch.cat(emb, dim=0)
            target_x0 = torch.cat(target_x0, dim=0)
        if target_emb.shape[1] > 0:
            if self.pos_emb_type == 'embedding':
                position_ids =self.position_ids[:target_emb.shape[1]]
            else:
                print('Not support non-embedding')
                assert 1==2
            target_emb = target_emb + self.pos_emb(position_ids)[None,:,:]
        return target_emb, target_x0 
