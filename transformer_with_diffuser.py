from __future__ import annotations
from transformer_only import *
import math
from math import sqrt
from typing import Literal
from functools import partial
from diffusers import UNet2DConditionModel, DDPMScheduler, CosineDPMSolverMultistepScheduler

import torch
from torch import nn, pi
from torch.special import expm1
import torch.nn.functional as F
from torch.nn import Module, ModuleList
import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
import os
from tqdm import tqdm
import pandas as pd

class AutoregressiveDiffusion(nn.Module):
    def __init__(
    self,
    max_seq_len,
    block_out_channels=(4, 8, 16, 16),
    cross_attention_dim=16,
    trans_dim=64,
    layers_per_block=2,
    norm_num_groups=4,
    heads=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dropout_p=0.05,
    device=None,
    diffusion_timestep=100,
    schedule_type='linear',
):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.device=device
        
        ## Transformer params.....
        self.dropout_p=dropout_p
        self.num_encoder_layers=num_encoder_layers
        self.num_decoder_layers=num_decoder_layers
        self.heads=heads
        self.dim_model=trans_dim
        # Causal transformer decoder (Ensures autoregressive training)
        self.transformer = CaloTransformer(dim_model=self.dim_model, num_heads=self.heads,
                                           num_encoder_layers=self.num_encoder_layers, 
                                           num_decoder_layers=self.num_decoder_layers, dropout_p=self.dropout_p).to(device)
        
        
       
        #diffusion params...
        self.num_train_timesteps=diffusion_timestep
        self.schedule_type= schedule_type
        if self.schedule_type=='cosine':
            self.scheduler =CosineDPMSolverMultistepScheduler(num_train_timesteps=self.num_train_timesteps)

        else:
            self.scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.schedule_type,
            beta_start=0.0001,  # default values, can adjust
            beta_end=0.02,
            clip_sample=True,
            prediction_type='epsilon'  # standard for DDPM
        )
        
        self.layers_per_block=layers_per_block
        self.block_out_channels=block_out_channels
        self.cross_attention_dim=cross_attention_dim
        self.norm_num_groups=norm_num_groups
        self.condition_proj = nn.Linear(1, self.cross_attention_dim)  # Project 1 → self.cross_attention_dim

        #customized UNet2DconditionModel
        self.denoiser = UNet2DConditionModel(
        sample_size=(self.max_seq_len,1),             # size of the generated image
        in_channels=1,              # input channels (e.g., latents or features)
        out_channels=1,             # output channels
        layers_per_block=self.layers_per_block,
        block_out_channels=self.block_out_channels,  # number of channels in each block
        down_block_types=(
           "DownBlock2D","DownBlock2D","CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
        up_block_types=(
             "CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D", "UpBlock2D",
        ),
        norm_num_groups=self.norm_num_groups,
        cross_attention_dim=self.cross_attention_dim  # size of the text embedding or conditioning vector
    )
        

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def forward(self, seq):
        """
        Training: Predict next token using past tokens (autoregressive)
        """
        #print("in training: ",seq[0].max(),seq[0].min(),seq[1].max(),seq[1].min())
        #l=(torch.randint(1, self.max_seq_len+1,(1,))).item()
        l=self.max_seq_len
        layers,Einc=seq[0][:,0:l,:],seq[1]
        
        input_seq = layers[:, :-1, :]
        target_seq = layers[:, 1:, :]

        tgt_mask = self.transformer.transformer.generate_square_subsequent_mask(input_seq.shape[1]).to(self.device)
        condition = self.transformer(Einc, input_seq, tgt_mask=tgt_mask)

        condition = self.condition_proj(condition)  # Now shape: (1024, 44, 16)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (target_seq.shape[0],), device=seq[0].device).long()
        noise = torch.randn_like(target_seq)
        noisy_x = self.scheduler.add_noise(target_seq, noise, timesteps).unsqueeze(1)
        noisy_x = noisy_x.to(self.device)
        timesteps = timesteps.to(self.device)
        condition = condition.to(self.device)
        noise_pred = self.denoiser(noisy_x, timesteps, encoder_hidden_states=condition).sample
        loss = nn.functional.mse_loss(noise_pred, noise.unsqueeze(1))

        return loss

    @torch.no_grad()
    def diffusion_step(self, cond_vector):
        self.denoiser.eval()
        ## for faster simulation: num_inference_steps=50
        #self.num_train_timesteps
        self.scheduler.set_timesteps(num_inference_steps=100)
        #print("cond_vector shape: ",cond_vector.shape)
        batch_size,seq_len, d = cond_vector.shape
        #cond = cond_vector.unsqueeze(1)  # (B, 1, D)
        cond = cond_vector
        x = torch.randn(batch_size, 1, 45, 1).to(cond.device) 
        prev_x = x.clone()

        for t in self.scheduler.timesteps:
            noise_pred = self.denoiser(x, t, encoder_hidden_states=cond).sample
            x = self.scheduler.step(noise_pred, t, x).prev_sample
            prev_x = x.clone()

        return x  # shape: (B, 45, 1)
    @torch.no_grad()
    def sampling_validate(self, incident_energies, num_layers=45):
        self.eval()
        B = incident_energies.shape[0]
        device = incident_energies.device

        generated = torch.zeros(B, 0, 1, device=device)

        for i in range(num_layers):
            tgt_mask = self.transformer.transformer.generate_square_subsequent_mask(generated.size(1) + 1).to(device)
            current_input = torch.cat([generated, torch.zeros(B, 1, 1, device=device)], dim=1)
            cond = self.transformer(incident_energies, current_input, tgt_mask=tgt_mask)
            cond = self.condition_proj(cond)
            out=self.diffusion_step(cond)
            out = out.squeeze(1)
            #print("shape of out: ",out.shape)
            next_val = out[:, -1:, :]
            #print("shape of next_val and generated: ",next_val.shape,generated.shape)
            generated = torch.cat([generated, next_val], dim=1)
        #print("shape of validation sampling: ", generated.shape)
        return generated
    
    @torch.no_grad()
    def sampling_val_loop(self,dataloader,device,epoch,model_dir):


        self.eval()
        generated = []
        reference = []
        energies = []
        num_samples=1000
        
        with torch.no_grad():
            sample_count = 0
            for batch in dataloader:
                # Adjust this depending on your dataloader structure
                real_shower, Einc = batch  # (B, 45, 1), (B, 1)
                batch_size = real_shower.shape[0]

                # Generate synthetic showers
                gen_shower = self.sampling_validate(Einc)

                generated.append(gen_shower.cpu())
                reference.append(real_shower.cpu())
                energies.append(Einc.cpu())

                sample_count += batch_size
                if sample_count >= num_samples:
                    break

        # Stack all outputs

        generated = torch.cat(generated, dim=0)[:num_samples]  # (N, 45, 1)
        reference = torch.cat(reference, dim=0)[:num_samples]  # (N, 45, 1)
        energies = torch.cat(energies, dim=0)[:num_samples]    # (N, 1)
        mse = torch.mean((generated - reference) ** 2)
        save_path = os.path.join(model_dir, f'generated_{epoch}.pt')
        ## need to update the path
        torch.save(generated,save_path)
        save_path = os.path.join(model_dir, f'reference_{epoch}.pt')
        torch.save(reference,save_path)
        return mse


    #not using this yet.
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
    
    # -------------------- Train Loop --------------------
    def train_loop(self,optimizer,  dataloader, device):
        self.train()
        total_loss = 0

        for batch in dataloader:
            
            layers, Einc = batch
            layers = layers.to(device)
            Einc = Einc.to(device)

            loss = self((layers, Einc))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    # -------------------- Validation Loop --------------------
    @torch.no_grad()
    def validation_loop(self, dataloader, device):
        self.eval()
        total_loss = 0

        for batch in dataloader:
            
            layers, Einc = batch
            layers = layers.to(device)
            Einc = Einc.to(device)

            loss = self((layers, Einc))

            #loss = loss_fn(pred, target_seq)
            total_loss += loss.item()

        return total_loss / len(dataloader)

def fit(model, optimizer, train_loader, val_loader, device, epochs,model_dir):
    loss_t=[]
    loss_v=[]
    MSE=[]
    best_val_r=float('inf')
    best_val_s=float('inf')
    model.to(device)
    min_e_r=-1
    min_e_s=-1
    os.makedirs(model_dir, exist_ok=True)
    summary_csv_path = './experiment_results_cosine/summary.csv'

    for epoch in range(epochs):
        train_loss = model.train_loop( optimizer,  train_loader, device)
        val_loss = model.validation_loop( val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
       
        mse=model.sampling_val_loop(val_loader, device, epoch,model_dir)
        loss_t.append(train_loss)
        loss_v.append(val_loss)
        MSE.append(mse)
        if best_val_r> val_loss:
            best_val_r = val_loss
            min_e_r=epoch
            save_path = os.path.join(model_dir, f'best_val_loss_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Saved model with best validation loss: {val_loss:.4f} at {save_path}")
        if best_val_s>mse:
            best_val_s = mse
            min_e_s=epoch
            save_path = os.path.join(model_dir, f'best_sampling_mse_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Saved model with best MSE for sampling validation: {mse:.4f} at {save_path}")
    
    config_name = os.path.basename(model_dir)  # Or parse from path if needed
    # Create a dictionary for this run
    result = {
        'config_name': config_name,
        'best_val_regular': best_val_r,
        'best_val_sampling': best_val_s,
        'regular_val_epoch':min_e_r,
        'sampling_val_epoch':min_e_s
    }
    
    # Check if CSV already exists
    if os.path.exists(summary_csv_path):
        # Load existing results
        df = pd.read_csv(summary_csv_path)
        # Append new result
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        # Create new DataFrame
        df = pd.DataFrame([result])

    # Save updated DataFrame
    df.to_csv(summary_csv_path, index=False)

    plot_losses(loss_t,loss_v,MSE,file_name='train_val_loss.pdf',file_path=model_dir)     

    