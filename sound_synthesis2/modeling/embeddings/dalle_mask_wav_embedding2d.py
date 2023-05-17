import torch
import torch.nn as nn
from .base_embedding import BaseEmbedding

class DalleMaskImageEmbedding(BaseEmbedding):
    def __init__(self,
                 num_embed=8192, # #should be quantize_number
                 spatial_size=[6, 100], # height and with 
                 embed_dim=3968, 
                 n_q = 10,
                 trainable=True,
                 pos_emb_type='embedding'
        ):
        super().__init__()
        
        if isinstance(spatial_size, int):
            spatial_size = [spatial_size, spatial_size]

        self.spatial_size = spatial_size
        self.num_embed = num_embed + 1 # add a mask token
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.pos_emb_type = pos_emb_type
        self.n_q = n_q

        assert self.pos_emb_type in ['embedding', 'parameter']
        
        self.emb = nn.Embedding(self.num_embed, embed_dim)
        if self.pos_emb_type == 'embedding':
            self.height_emb = nn.Embedding(self.spatial_size[0], embed_dim) # height   
            self.width_emb = nn.Embedding(self.spatial_size[1], embed_dim) # width
        else:
            self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim)) # height #32,1024
            self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim)) # width   #32,1024
        self._set_trainable()

    def forward(self, index, **kwargs):
        # print('index ',index.shape)
        # assert 1==2
        assert index.dim() == 2 # B x L
        try:
            index[index < 0] = 0  
            emb = self.emb(index)
        except:
            raise RuntimeError('IndexError: index out of range in self, max index {}, num embed {}'.format(index.max(), self.num_embed))
        # add col and row embedding
        if emb.shape[1] > 0:
        # if False:
            H = self.n_q
            W = index.shape[1]//self.n_q
            if self.pos_emb_type == 'embedding':
                height_emb = self.height_emb(torch.arange(H, device=index.device).view(1, H)).unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
                # print('height_emb ', height_emb.shape)
                width_emb = self.width_emb(torch.arange(W, device=index.device).view(1, W)).unsqueeze(1) # 1 x W x D -> 1 x 1 x W x D
                # print('width_emb ', width_emb.shape)
            else:
                print('self.pos_emb_type must be embedding')
                assert 1==2
                height_emb = self.height_emb.unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
                width_emb = self.width_emb.unsqueeze(1) # 1 x W x D -> 1 x 1 x W x D
            pos_emb = (height_emb + width_emb).view(1, H * W, -1) # 1 x H x W x D -> 1 x L xD
            # print('pos_emb ', pos_emb.shape)
            emb = emb + pos_emb[:, :emb.shape[1], :]
        return emb
