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
        self.emb = nn.Embedding(self.num_embed, embed_dim) # 
        self.register_buffer('position_ids', torch.arange(max_size))
        if self.pos_emb_type == 'embedding':
            self.pos_emb = nn.Embedding(self.max_size, embed_dim)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_size, embed_dim))
        self._set_trainable()

    def forward(self, index, x_0, s, **kwargs):
        assert index.dim() == 2 # B x L
        try:
            index[index < 0] = 0  
            emb = self.emb(index)
            x_0[x_0<0] = 0
            x_0_emb = self.emb(x_0.long())
        except:
            raise RuntimeError('IndexError: index out of range in self, max index {}, num embed {}'.format(index.max(), self.num_embed))
        # adding position enbedding
        if index.shape[1] == x_0.shape[1]:
            emb = emb.reshape(emb.shape[0], self.n_q, -1, emb.shape[-1])
            x_0_emb = x_0_emb.reshape(x_0_emb.shape[0], self.n_q, -1, x_0_emb.shape[-1])
            target_emb = []
            target_x0 = []
            for b in range(emb.shape[0]):
                tmp_x0 = torch.zeros_like(x_0_emb[0,0,:,:]) # all zero
                target_emb.append(emb[b,s[b],:,:].unsqueeze(0))
                for i in range(s[b]):
                    tmp_x0 += x_0_emb[b,i,:,:]
                target_x0.append(tmp_x0.unsqueeze(0))
            target_emb = torch.cat(target_emb,dim=0)
            target_x0 = torch.cat(target_x0, dim=0)
        else:
            x_0_emb = x_0_emb.reshape(x_0_emb.shape[0], self.n_q, -1, x_0_emb.shape[-1])
            target_x0 = []
            for b in range(x_0_emb.shape[0]):
                tmp_x0 = torch.zeros_like(x_0_emb[0,0,:,:]) # all zero
                #print('tmp_x0 ', tmp_x0.shape)
                #print(emb[b,s[b],:,:].unsqueeze(0).shape)
                for i in range(s[b]):
                    tmp_x0 += x_0_emb[b,i,:,:]
                target_x0.append(tmp_x0.unsqueeze(0))
            target_emb = emb
            target_x0 = torch.cat(target_x0, dim=0)
        
        if target_emb.shape[1] > 0:
            if self.pos_emb_type == 'embedding':
                # print('self.position_ids ', self.position_ids.shape)
                # print('emb.shape[1] ', emb.shape[1])
                position_ids =self.position_ids[:target_emb.shape[1]]
            else:
                print('Not support non-embedding')
                assert 1==2
            # print('position_ids ', position_ids)
            # print('emb ', emb.shape)
            # print('self.pos_emb ', self.pos_emb)
            # assert 1==2
            target_emb = target_emb + self.pos_emb(position_ids)[None,:,:]
        return target_emb, target_x0 
