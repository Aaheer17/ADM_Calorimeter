from __future__ import annotations

import math
from math import sqrt
from typing import Literal
from functools import partial
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from torch import nn, pi
from torch.special import expm1
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from conditional_unet import *
import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from x_transformers import Decoder

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(num, den, eps = 1e-5):
    return num / den.clamp(min = eps)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim

    if padding_dims <= 0:
        return t

    return t.view(*t.shape, *((1,) * padding_dims))


class AutoregressiveDiffusion(nn.Module):
    def __init__(
        self,
        dim,
        *,
        max_seq_len,
        depth=8,
        dim_head=64,
        heads=8,
        mlp_depth=3,
        mlp_width=None,
        dim_input=None,
        decoder_kwargs=dict(),
        mlp_kwargs=dict(),
        diffusion_kwargs=dict(clamp_during_sampling=True)
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.dim_input = dim if dim_input is None else dim_input

        
        # Absolute positional embeddings
        print(f"shape of dim input {self.dim_input} and dim {dim}")
        self.num_train_timesteps=100
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)
        self.scheduler = DDPMScheduler(self.num_train_timesteps)
        # Input projection
        self.proj_in = nn.Linear(self.dim_input, dim)

        # Causal transformer decoder (Ensures autoregressive training)
        self.transformer = Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dim_head=dim_head,
            **decoder_kwargs,
            
        )

        #customized UNet2DconditionModel
        self.denoiser = UNet2DConditionModel(
        sample_size=(self.max_seq_len,1),             # size of the generated image
        in_channels=1,              # input channels (e.g., latents or features)
        out_channels=1,             # output channels
        layers_per_block=2,
        block_out_channels=(4,8,16, 16),  # number of channels in each block
        down_block_types=(
           "DownBlock2D","DownBlock2D","CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
        up_block_types=(
             "CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D", "UpBlock2D",
        ),
        norm_num_groups=4,
        cross_attention_dim=16  # size of the text embedding or conditioning vector
    )
        

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, seq):
        """
        Training: Predict next token using past tokens (autoregressive)
        """
        #print("in training: ",seq[0].max(),seq[0].min(),seq[1].max(),seq[1].min())
        #l=(torch.randint(1, self.max_seq_len+1,(1,))).item()
        l=self.max_seq_len
        layer_seq,Einc=seq[0][:,0:l,:],seq[1]
        
        b, seq_len, dim = layer_seq.shape
        
        Einc = self.proj_in(Einc.view(b, 1, -1))
      
        assert dim == self.dim_input
        # assert seq_len == self.max_seq_len

        seq, target = layer_seq[:, :-1], layer_seq #main jhamela. see betar data loader

        # Project input tokens
        #seq = self.proj_in(seq)
      
        #seq = torch.cat((Einc, seq), dim=1)
       
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (target.shape[0],), device=seq.device).long()
        noise = torch.randn_like(target)
        noisy_x = self.scheduler.add_noise(target, noise, timesteps).unsqueeze(1)
     
        # Add positional embeddings
#         seq += self.abs_pos_emb(torch.arange(seq.shape[1], device=self.device)) #seq+= keno

#         cond = self.transformer(seq)
        
        noise_pred = self.denoiser(noisy_x, timesteps, encoder_hidden_states=Einc).sample
        #print("shape of noise_pred: ",noise_pred.shape)

        loss = nn.functional.mse_loss(noise_pred, noise.unsqueeze(1))

        return loss

    @torch.no_grad()
    def diffusion_step(self, cond_vector, num_steps=100):
        self.denoiser.eval()
        self.scheduler.set_timesteps(num_steps)

        batch_size, d = cond_vector.shape
        cond = cond_vector.unsqueeze(1)  # (B, 1, D)

        x = torch.randn(batch_size, 1, 45, 1).to(cond.device) 
        prev_x = x.clone()

        for t in self.scheduler.timesteps:
            noise_pred = self.denoiser(x, t, encoder_hidden_states=cond).sample
            x = self.scheduler.step(noise_pred, t, x).prev_sample
            prev_x = x.clone()

        return x  # shape: (B, 45, 1)

    
    @torch.no_grad()
    def sample(self, batch_size=1, incident_energy=None):
        """
        Generate an entire shower sample in one shot conditioned on Einc.

        Args:
            batch_size: Number of showers to generate.
            incident_energy: Tensor of shape (B, 1)

        Returns:
            Tensor of shape (B, 45, 1)
        """
        assert incident_energy is not None, "incident_energy must be provided"
        self.eval()

        # Project Einc if needed
        cond = self.proj_in(incident_energy)  # (B, D)

        # Generate full shower
        print("shape of cond before step: ",cond.shape)
        generated = self.diffusion_step(cond)  # (B, 45, 1)
        print("shape of generated: ",generated.shape)
        return generated.squeeze(1)


# image wrapper

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5