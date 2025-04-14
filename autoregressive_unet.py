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

        # Start token for sequence generation
        #self.start_token = nn.Parameter(torch.zeros(dim))#check nn.parameter

        # Absolute positional embeddings
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)
        self.scheduler = DDPMScheduler(num_train_timesteps=100)
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

        # MLP-based denoiser
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
        # self.denoiser._init_weights()
        # added by Farzana
    #     dim_cond=dim   #better fix required
    #     ### Unet based denoiser
        #dim_cond=dim
        

        # Diffusion model for denoising
        # self.diffusion = ElucidatedDiffusion(
        #     self.dim_input,
        #     self.denoiser,
        #     **diffusion_kwargs,
        #     dim_cond=dim_cond
        # )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, seq):
        """
        Training: Predict next token using past tokens (autoregressive)
        """
        layer_seq,Einc=seq[0],seq[1]
        #print(layer_seq[0:2])
        b, seq_len, dim = layer_seq.shape
        #Einc = Einc.view(Einc.shape[0], 1, self.dim_input)  # Shape: (batch, 1, dim_input)
        torch.save(Einc, "data_debug/Einc_before.pt")
        
        Einc = self.proj_in(Einc.view(b, 1, -1))
        torch.save(Einc, "data_debug/Einc_after.pt")
        
        b, seq_len, dim = layer_seq.shape
        assert dim == self.dim_input
        assert seq_len == self.max_seq_len

        # Shift target right for autoregressive learning
        # target = seq[:, 1:]  # Next token
        # seq = seq[:, :-1]  # Past tokens as input
        
        seq, target = layer_seq[:, :-1], layer_seq #main jhamela. see betar data loader
        torch.save(seq, "data_debug/seq_before_project.pt")
        torch.save(target, "data_debug/target_no_projection.pt")

        # Project input tokens
        seq = self.proj_in(seq)
        #target=self.proj_in(target)
        # Append start token at the beginning
        #start_token = repeat(self.start_token, 'd -> b 1 d', b=b)
        #print("shape of Einc and seq: ",Einc.shape, seq.shape)
        seq = torch.cat((Einc, seq), dim=1)
        #print("self.scheduler.config.num_train_timesteps: ",self.scheduler.config.num_train_timesteps)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (target.shape[0],), device=seq.device).long()
        noise = torch.randn_like(target)
        noisy_x = self.scheduler.add_noise(target, noise, timesteps).unsqueeze(1)

        # Add positional embeddings
        seq += self.abs_pos_emb(torch.arange(seq.shape[1], device=self.device)) #seq+= keno

        # Initialize loss
        #loss = 0
        
        cond = self.transformer(seq)
        #print("shape of the condition after passing through Transformer: ",cond.shape )
        # pack batch and sequence dimensions, so to train each token with different noise levels

        # target, _ = pack_one(target, '* d')
        # cond, _ = pack_one(cond, '* d')
        #print(f"shape of the condition after not passing through pack one {cond.shape}, noisy_x shape: {noisy_x.shape}, timesteps: {timesteps.shape} " )
        noise_pred = self.denoiser(noisy_x, timesteps, encoder_hidden_states=cond).sample

        loss = nn.functional.mse_loss(noise_pred, noise.unsqueeze(1))

        

        return loss


#         # Autoregressive Training Loop
#         for t in range(seq.shape[1] - 1):  # Exclude last timestep
#             current_input = seq[:, :t+1]  # Use only past tokens
#             cond = self.transformer(current_input)  # Transformer encoding
#             last_cond = cond[:, -1]  # Get last token's encoding

#             # Compute diffusion loss
#             loss += self.diffusion(target[:, t], cond=last_cond)

#         return loss / (seq.shape[1] - 1)  # Normalize loss

    
    @torch.no_grad()
    def diffusion_step(self, cond_vector, num_steps=100):
        """
        Perform diffusion-based denoising for one step.

        Args:
            cond_vector: Tensor of shape (B, D), i.e., last transformer hidden state

        Returns:
            Tensor of shape (B, 1, 1) — the next layer to append
        """
        self.denoiser.eval()
        self.scheduler.set_timesteps(num_steps)

        batch_size, d = cond_vector.shape
        cond = cond_vector.unsqueeze(1)  # shape: (B, 1, D)

        # Start from noise
        x = torch.randn(batch_size, 1, 1, 1).to(cond.device)  # (B, 1, H=1, W=1)

        for t in self.scheduler.timesteps:
            noise_pred = self.denoiser(x, t, encoder_hidden_states=cond).sample
            x = self.scheduler.step(noise_pred, t, x).prev_sample

        return x.view(batch_size, 1, 1)  # return (B, 1, 1)
    @torch.no_grad()
    def sample(
        self,
        batch_size=1,
        incident_energy=None,
        prompt=None
    ):
        self.eval()
        assert incident_energy is not None, "incident_energy must be provided"

        incident_energy = incident_energy.view(batch_size, 1, self.dim_input)  # (B, 1, D)
        is_normalized = ((incident_energy >= -1.0) & (incident_energy <= 1.0)).all()
        print("Is normalized in [-1, 1]? ->", is_normalized.item(),incident_energy.min(),incident_energy.max())
        out = incident_energy

        cache = None

        for _ in tqdm(range(self.max_seq_len), desc="Generating layers"):
            cond = self.proj_in(out)
            cond = cond + self.abs_pos_emb(torch.arange(cond.shape[1], device=self.device))
            cond, cache = self.transformer(cond, cache=cache, return_hiddens=True)

            last_cond = cond[:, -1]  # shape: (B, D)

            denoised_pred = self.diffusion_step(last_cond)  # shape: (B, 1, 1)
            out = torch.cat((out, denoised_pred), dim=1)

        return out  # shape: (B, 46, 1)

    

    

# image wrapper

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5