import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from torchdiffeq import odeint
from typing import Optional
import inspect

class energyTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.dim_embedding = self.params["dim_embedding"]
        self.dims_in = self.params["shape"][0]
        self.dims_c = self.params["n_con"]
        self.bayesian = False
        self.layer_cond = self.params.get("layer_cond", False)

        self.c_embed = self.params.get("c_embed", None)
        self.x_embed = self.params.get("x_embed", None)

        self.encode_t_dim = self.params.get("encode_t_dim", 64)
        self.encode_t_scale = self.params.get("encode_t_scale", 30)
        #print("in ARTRansformer: ",self.c_embed,self.x_embed, self.encode_t_dim, self.encode_t_scale)
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=params["n_head"],
            num_encoder_layers=params["n_encoder_layers"],
            num_decoder_layers=params["n_decoder_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params.get("dropout_transformer", 0.0),
            # activation=params.get("activation", "relu"),
            batch_first=True,
        )
        if self.x_embed:
            #nn.Linear(1, self.dim_embedding) changed by Farzana, age 1 silo
            self.x_embed = nn.Sequential(
                nn.Linear(44, self.dim_embedding),#changed 44 from 1
                nn.Linear(self.dim_embedding, self.dim_embedding)
            )
        if self.c_embed:
            self.c_embed = nn.Sequential(
                nn.Linear(1, self.dim_embedding),
                nn.ReLU(),
                nn.Linear(self.dim_embedding, self.dim_embedding)
            )
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.encode_t_dim, scale=self.encode_t_scale),
            nn.Linear(self.encode_t_dim, self.encode_t_dim)
        )
        
        
        
    def compute_embedding(
        self, p: torch.Tensor, dim: int, embedding_net: Optional[nn.Module]
    ) -> torch.Tensor:
        
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        # print("In compute_embedding ", inspect.stack()[1])
        print("dimension of p: ", p.size())
        one_hot = torch.eye(dim, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        #print("self.embed_net ",embedding_net, one_hot.size())
        if embedding_net is None:
            n_rest = self.dim_embedding - dim - p.shape[-1]
            #print(f"what is n_rest {n_rest}")
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            #print("zeros, p and one_hot shape: ",zeros.size(),p.shape,one_hot.shape)
            return torch.cat((p, one_hot, zeros), dim=2)
        else:
            #print(f"compute {p.size()}")
            return self.positional_encoding(embedding_net(p))
        
        
        
    def forward(self, c, t=None, x=None, rev=False):
        #print("in forward OF aRTRANFORMER ")
        # print("Calling function:", inspect.stack()[1].function)
        # print("Calling file:", inspect.stack()[1].filename)
        
        # print("Calling line:", inspect.stack()[0][3])
        # print(" in ARtransformer forward: ",c.shape)
        if not rev:
           
            xp = nn.functional.pad(x[:, :-1], (0, 0, 1, 0))
            #print(f"in if {x.device}, {xp.shape} ,{c.shape}, {xp.shape}, {self.dims_c}, {self.dims_in},{self.c_embed}, {self.x_embed}")
            embedding = self.transformer(
                src=self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed),
                tgt=self.compute_embedding(xp, dim=self.dims_in + 1, embedding_net=self.x_embed),
                tgt_mask=torch.ones(
                    (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )

            if self.layer_cond:
                
                layer_one_hot = repeat(
                    torch.eye(self.dims_in, device=x.device), '... -> b ...', b=len(c)
                )
                #print("insider_cond ")
                embedding = torch.cat([embedding, layer_one_hot], dim=2)

            t = self.t_embed(t)
            #print("t  and embedding er shape after embedding : ",t.shape,embedding.shape)
            return embedding,t
            #pred = self.subnet(torch.cat([x_t, t, embedding], dim=-1))

        else:
            #print("shape of x and batch size: ", x.shape, c.shape[0])
            batch_size = c.shape[0]
            device = c.device
            dtype = c.dtype

            # Ensure `x_prev` is correctly initialized
            if x is None or x.shape[1] == 0:
                x = torch.zeros((batch_size, 1, 1), device=device, dtype=dtype)
            else:
                x = torch.cat([x, torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)], dim=1)

            # Compute conditioning embedding
            c_embed = self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed)
            

            # Ensure `x_embed` has the right dimensions
            seq_len = max(1, x.shape[1])
            x_embed = self.compute_embedding(x, dim=seq_len, embedding_net=self.x_embed)

            # Ensure target mask has the right dimensions
            tgt_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).triu(diagonal=1)

            # Transformer processing
            embedding = self.transformer(src=c_embed, tgt=x_embed, tgt_mask=tgt_mask)
            return embedding
            # Time embedding
            # if t is not None:
            #     t_embed = self.t_embed(t)
            #     return embedding, t_embed
            # else:
            #     return embedding,None

            #print("for sampling before returning: ", embedding.shape, t_embed.shape)

            

    def time_embedding(self, t):
        return self.t_embed(t)

    #embeddings = []
#             x = torch.zeros((c.shape[0], 1, 1), device=c.device, dtype=c.dtype)
#             print("sampling error: ",c.shape, self.dims_in)
#             c_embed = self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed)
#             for i in range(self.dims_in):
#                 embedding = self.transformer(
#                     src=c_embed,
#                     tgt=self.compute_embedding(x, dim=self.dims_in + 1, embedding_net=self.x_embed),
#                     tgt_mask=torch.ones(
#                         (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
#                     ).triu(diagonal=1),
#                 )
#                 if self.layer_cond:
#                     print("self.layer_cond: ",self.layer_cond)
#                     layer_one_hot = repeat(
#                         F.one_hot(torch.tensor(i, device=x.device), self.dims_in),
#                         'd -> b 1 d', b=len(c)
#                     )
#                     embedding = torch.cat([embedding[:, -1:,:], layer_one_hot], dim=2)
#                 # x_new = self.sample_dimension(embedding[:, -1:, :])
#                 # x = torch.cat((x, x_new), dim=1)
                
#             return embedding,t
            
            # pred = x[:, 1:]
            # pred = pred.squeeze()
        #print("pred er size ",pred.shape)
        



class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)