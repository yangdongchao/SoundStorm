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
        self.embs = nn.ModuleDict({
            '0': nn.Embedding(self.num_embed, embed_dim),
            '1': nn.Embedding(self.num_embed, embed_dim),
            '2': nn.Embedding(self.num_embed, embed_dim),
            '3': nn.Embedding(self.num_embed, embed_dim),
            '4': nn.Embedding(self.num_embed, embed_dim),
            '5': nn.Embedding(self.num_embed, embed_dim),
            '6': nn.Embedding(self.num_embed, embed_dim),
            '7': nn.Embedding(self.num_embed, embed_dim)
        }) 
        self.register_buffer('position_ids', torch.arange(max_size))
        if self.pos_emb_type == 'embedding':
            self.pos_emb = nn.Embedding(self.max_size, embed_dim)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_size, embed_dim))
        self._set_trainable()

    def forward(self, index, **kwargs):
        assert index.dim() == 2 # B x L
        # print('index ', index.shape)
        index = index.reshape(index.shape[0], self.n_q, -1) # 
        index[index < 0] = 0
        emb = []
        for b in range(index.shape[0]):
            sub_emb = []
            for i in range(self.n_q):
                tmp_emb = self.embs[str(i)](index[b,i,:])
                sub_emb.append(tmp_emb.unsqueeze(0))
            sub_emb = torch.cat(sub_emb, dim=0) # n_q, len, dim
            emb.append(sub_emb.unsqueeze(0))
        target_emb = torch.cat(emb, dim=0)
        # print('emb3 ', emb.shape)
        #print('target_emb ', target_emb.shape)
        if target_emb.shape[2] > 0:
            if self.pos_emb_type == 'embedding':
                # print('self.position_ids ', self.position_ids.shape)
                # print('emb.shape[1] ', emb.shape[1])
                position_ids =self.position_ids[:target_emb.shape[2]]
            else:
                print('Not support non-embedding')
                assert 1==2
            #emb = emb + self.pos_emb(position_ids)[None,:,:]
        return target_emb, self.pos_emb(position_ids)[None,:,:]
