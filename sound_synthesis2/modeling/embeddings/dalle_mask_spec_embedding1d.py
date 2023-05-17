import torch
import torch.nn as nn
from .base_embedding import BaseEmbedding

class DalleMaskImageEmbedding(BaseEmbedding):
    def __init__(self,
                 num_embed=8192, # #should be quantize_number
                 max_size= 1000, # height and with 
                 embed_dim=3968, 
                 trainable=True,
                 pos_emb_type='embedding'
        ):
        super().__init__()
        self.max_size = max_size
        self.num_embed = num_embed + 1 # add a mask token
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.pos_emb_type = pos_emb_type
        assert self.pos_emb_type in ['embedding', 'parameter']
        self.emb = nn.Embedding(self.num_embed, embed_dim)
        self.register_buffer('position_ids', torch.arange(max_size))
        if self.pos_emb_type == 'embedding':
            self.pos_emb = nn.Embedding(self.max_size, embed_dim)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_size, embed_dim))
        self._set_trainable()

    def forward(self, index, **kwargs):
        assert index.dim() == 2 # B x L
        try:
            index[index < 0] = 0  
            emb = self.emb(index)
        except:
            raise RuntimeError('IndexError: index out of range in self, max index {}, num embed {}'.format(index.max(), self.num_embed))
        # adding position enbedding
        if emb.shape[1] > 0:
            if self.pos_emb_type == 'embedding':
                # print('self.position_ids ', self.position_ids.shape)
                # print('emb.shape[1] ', emb.shape[1])
                position_ids =self.position_ids[:emb.shape[1]]
            else:
                print('Not support non-embedding')
                assert 1==2
            # print('position_ids ', position_ids)
            # print('emb ', emb.shape)
            # print('self.pos_emb ', self.pos_emb)
            # assert 1==2
            emb = emb + self.pos_emb(position_ids)[None,:,:]
        return emb
