from __future__ import annotations
from typing import Literal
from diffusers import UNet2DConditionModel, DDPMScheduler,CosineDPMSolverMultistepScheduler,DDIMScheduler
import torch
from torch import nn, pi
from torch.special import expm1
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from conditional_unet import *
import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from utils import *
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
class DenoisingDiffusion(nn.Module):
    def __init__(
        self,
        max_seq_len,
        block_out_channels=(4, 8, 16, 16),
        cross_attention_dim=16,
       
        layers_per_block=2,
        norm_num_groups=4,
        
        device=None,
        diffusion_timestep=100,
        schedule_type='DDPM',
        prediction_type='sample',
        num_inference_steps=100
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.device=device
        #diffusion params...
        self.num_train_timesteps=diffusion_timestep
        self.schedule_type= schedule_type
        self.num_inference_steps=num_inference_steps
        self.prediction_type=prediction_type # e.g., 'v_prediction', 'epsilon', or 'sample'
                        
        if self.schedule_type=='cosine':
            self.scheduler = CosineDPMSolverMultistepScheduler(
                            num_train_timesteps=self.num_train_timesteps,  # Total training steps for diffusion
                            prediction_type=self.prediction_type           # e.g., 'v_prediction', 'epsilon', 
                        )
            ## IGNORE COSINE SCHEDULER...
        elif self.schedule_type=='DDIM':
            self.scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps, prediction_type=self.prediction_type)
        elif self.schedule_type=='DDPM':
            self.scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.schedule_type,
            beta_start=0.0001,  # default values, can adjust
            beta_end=0.02,
            clip_sample=True,
            prediction_type=self.prediction_type  # standard for DDPM
        )
        else:
            print(f"Not implemented for schedule type: {self.schedule_type}")
            
        self.layers_per_block=layers_per_block
        self.block_out_channels=block_out_channels
        self.cross_attention_dim=cross_attention_dim
        self.norm_num_groups=norm_num_groups
        self.condition_proj = nn.Linear(1, self.cross_attention_dim)  # Project 1 → self.cross_attention_dim
        
        #Customized UNet2DconditionModel
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

    def forward(self, seq, seq_length,frob_loss=False):
        """
        Training: Predict next token using past tokens (autoregressive)
        """
        #seq_length should be less than 44, when seq_length>0 we will see partial layers during training
        if seq_length>0:
            l=seq_length
        else: 
            l=self.max_seq_len
        layer_seq,Einc=seq[0][:,0:l,:],seq[1]
        b, seq_len, dim = layer_seq.shape
        Einc = self.condition_proj(Einc.view(b, 1, -1))  # passed through embedder
        target = layer_seq 
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (target.shape[0],), device=self.device).long()
        noise = torch.randn_like(target)
        noisy_x = self.scheduler.add_noise(target, noise, timesteps).unsqueeze(1)
        noise_pred = self.denoiser(noisy_x, timesteps, encoder_hidden_states=Einc).sample
     
        if self.prediction_type=='epsilon':
            loss = nn.functional.mse_loss(noise_pred, noise.unsqueeze(1))
        elif self.prediction_type=='sample':
            loss = nn.functional.mse_loss(noise_pred, target.unsqueeze(1))
            if frob_loss:
                frob_dist=calculate_frobenius_distance(noise_pred,target)
                loss+=0.01*frob_dist
        elif self.prediction_type == 'v_prediction':
            # Get alphas and sigmas from the scheduler
            alphas_cumprod = self.scheduler.alphas_cumprod[timesteps].reshape(-1, 1, 1).to(noise.device)
            sqrt_alpha = alphas_cumprod.sqrt()
            sqrt_one_minus_alpha = (1 - alphas_cumprod).sqrt()

            # Compute v_target
            v_target = sqrt_alpha * noise.unsqueeze(1) - sqrt_one_minus_alpha * target.unsqueeze(1)

            # Compute loss
            loss = nn.functional.mse_loss(noise_pred, v_target)
        else:
            raise ValueError(f"Invalid prediction_type: {self.prediction_type}")
            
        return loss

    @torch.no_grad()
    def diffusion_step(self, cond):
        self.denoiser.eval()
        self.scheduler.set_timesteps(self.num_inference_steps)
        batch_size = cond.shape[0]  # Get batch size from cond tensor
        x = torch.randn(batch_size, 1, self.max_seq_len, 1).to(cond.device)

        for t in self.scheduler.timesteps:
            noise_pred = self.denoiser(x, t, encoder_hidden_states=cond).sample
            x = self.scheduler.step(noise_pred, t, x).prev_sample

        return x.squeeze(1)  # shape: (B, self.max_seq_len, 1)

    @torch.no_grad()
    def sample(self, incident_energy=None):
        """
        Generate an entire shower sample in one shot conditioned on Einc.
        Args:
            batch_size: Number of showers to generate.
            incident_energy: Tensor of shape (B, 1)
        Returns:
            Tensor of shape (B, self.max_seq_len, 1)
        """
        assert incident_energy is not None, "incident_energy must be provided"
        incident_energy = incident_energy.to(self.device)
        self.eval()
        # Project Einc if needed
        cond = self.condition_proj(incident_energy.view(incident_energy.shape[0], 1, -1))  # Project conditioning
        # Generate full shower
        print("shape of cond before step: ", cond.shape)
        generated = self.diffusion_step(cond)  # (B, self.max_seq_len, 1)
        print("shape of generated: ", generated.shape)
        return generated
    
    def train_loop(self, optimizer, dataloader,seq_length,frob_loss):
        self.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            layers, Einc = batch
            layers = layers.to(self.device)
            Einc = Einc.to(self.device)
            loss = self((layers, Einc),seq_length=seq_length,frob_loss=frob_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validation_loop(self, dataloader,seq_length=0,frob_loss=False):
        self.eval()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Validation"):
            layers, Einc = batch
            layers = layers.to(self.device)
            Einc = Einc.to(self.device)
            loss = self((layers, Einc),seq_length=seq_length,frob_loss=frob_loss)
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def sampling_val_loop(self, dataloader, device, epoch, model_dir,n_samples):
        self.eval()
        total_mse = 0
        count = 0
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        sampling_times = []
        real_all = []
        generated_all = []
        Einc_all=[]
        model_dir = os.path.dirname(model_dir) if model_dir.endswith('.pt') else model_dir
        # Create directory for saving sample visualizations
        samples_dir = os.path.join(model_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        print("sampling directory: ",samples_dir)
        
        
        for batch in tqdm(dataloader, desc="Sampling Validation"):
            layers, Einc = batch
            layers = layers.to(device)
            Einc = Einc.to(device)
            batch_size=Einc.shape[0]
            # Time the sampling process
            start_time.record()
            generated = self.sample(incident_energy=Einc)
            end_time.record()
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            elapsed_time = start_time.elapsed_time(end_time)
            sampling_times.append(elapsed_time)
            if epoch=='test':
                # Track all data for saving
                real_all.append(layers.cpu())
                generated_all.append(generated.cpu())
                Einc_all.append(Einc.cpu())

            # Calculate MSE between generated and ground truth
            mse = F.mse_loss(generated, layers).item()
            total_mse += mse
            count += batch_size
            if count>=n_samples:
                break
            
            
            # Save a few sample visualizations (e.g., first batch of each epoch)
            # if count == 1:
            #     self._save_sample_visualization(layers, generated, epoch, samples_dir)
        
        avg_sampling_time = (sum(sampling_times) / len(sampling_times))/1000
        avg_mse = total_mse / count
        
        # Concatenate all generated and real samples
        if epoch=='test':
            real_all = torch.cat(real_all, dim=0)
            generated_all = torch.cat(generated_all, dim=0)
            Einc_all=torch.cat(Einc_all,dim=0)
            save_path = os.path.join(samples_dir, f"all_generated_vs_real_epoch_{epoch}.pt")
            torch.save({
                'generated': generated_all,
                'real': real_all,
                'Einc':Einc_all,
            }, save_path)
            frob_dist=calculate_frobenius_distance(generated_all,real_all)
             # Record experiment results
            config_name = os.path.basename(model_dir)
            result = {
                'config_name': config_name,
                'frob_dist':frob_dist,
                'avg_sampling_time(per batch in seconds)':avg_sampling_time,
                'avg_mse':avg_mse
            }
            summary_csv_path = os.path.join(os.path.dirname(model_dir), 'test.csv')
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
            print(f"✓ Saved all test samples to {save_path}")

        print(f"Sampling Validation - MSE: {avg_mse:.4f}, Avg Sampling Time: {avg_sampling_time:.2f}ms")
        return avg_mse, avg_sampling_time
    
    def _save_sample_visualization(self, ground_truth, generated, epoch, samples_dir):
        """Save visualization of ground truth vs generated samples"""
        # Take the first sample from the batch for visualization
        gt = ground_truth[0].cpu().numpy()
        gen = generated[0].cpu().numpy()
        
        plt.figure(figsize=(10, 5))
        
        # Plot ground truth
        plt.subplot(1, 2, 1)
        plt.imshow(gt.T, aspect='auto', cmap='viridis')
        plt.title('Geant4')
        plt.colorbar()
        
        # Plot generated sample
        plt.subplot(1, 2, 2)
        plt.imshow(gen.T, aspect='auto', cmap='viridis')
        plt.title('Generated')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f'sample_epoch_{epoch}.png'))
        plt.close()
    

def fit(model, optimizer, train_loader, val_loader, device, epochs, model_dir,frob_loss=False,seq_length=0):
    """Train the diffusion model and evaluate performance"""
    loss_t = []
    loss_v = []
    MSE = []
    best_val_r = float('inf')
    best_val_s = float('inf')
    model.to(device)
    min_e_r = -1
    min_e_s = -1
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    summary_csv_path = os.path.join(os.path.dirname(model_dir), 'summary.csv')
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train and validate
        train_loss = model.train_loop(optimizer, train_loader, seq_length, frob_loss)
        val_loss = model.validation_loop(val_loader, seq_length, frob_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Run sampling validation
        mse, sampling_time = model.sampling_val_loop(val_loader, device, epoch, model_dir,n_samples=1000)
        
        # Store metrics
        loss_t.append(train_loss)
        loss_v.append(val_loss)
        MSE.append(mse)
        
        # Save best model based on validation loss
        if best_val_r > val_loss:
            best_val_r = val_loss
            min_e_r = epoch
            save_path = os.path.join(model_dir, f'best_val_loss_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved model with best validation loss: {val_loss:.4f} at {save_path}")
        
        # Save best model based on sampling MSE
        if best_val_s > mse:
            best_val_s = mse
            min_e_s = epoch
            save_path = os.path.join(model_dir, f'best_sampling_mse_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved model with best MSE for sampling validation: {mse:.4f} at {save_path}")
        
        # Save checkpoint every n epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'sampling_mse': mse
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at {checkpoint_path}")
    
    # Record experiment results
    config_name = os.path.basename(model_dir)
    result = {
        'config_name': config_name,
        'best_val_regular': best_val_r,
        'best_val_sampling': best_val_s,
        'regular_val_epoch': min_e_r,
        'sampling_val_epoch': min_e_s
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
    
    # Plot and save loss curves
    plot_losses(loss_t, loss_v, MSE, file_name='train_val_loss.pdf', file_path=model_dir)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_r:.4f} at epoch {min_e_r+1}")
    print(f"Best sampling MSE: {best_val_s:.4f} at epoch {min_e_s+1}")
    
    #return model

def calculate_frobenius_distance(
    generated_all: torch.Tensor,
    real_all: torch.Tensor,
   
):
    """
    Compute Frobenius distance between correlation matrices of generated and real samples
    
    Args:
        generated_all (torch.Tensor): Generated samples, shape (N, 45, 1)
        real_all (torch.Tensor): Real samples, shape (N, 45, 1)
        sampling_time (float): Average sampling time in ms
        config_idx (int): Identifier for config/run
        save_path (str): Path to save the CSV file
    """
     # If input is 4D (e.g., [N, 1, 45, 1]), squeeze the second dimension
    if generated_all.ndim == 4:
        generated_all = generated_all.squeeze(1)
    if real_all.ndim == 4:
        real_all = real_all.squeeze(1)
    # Squeeze the last dimension → shape: (N, 45)
    gen_sample = generated_all.squeeze(-1).numpy() if isinstance(generated_all, torch.Tensor) else generated_all.squeeze(-1)
    ref_sample = real_all.squeeze(-1).numpy() if isinstance(real_all, torch.Tensor) else real_all.squeeze(-1)

    # Compute correlation matrices
    corr_gen = np.corrcoef(gen_sample, rowvar=False)
    corr_ref = np.corrcoef(ref_sample, rowvar=False)

    # Compute Frobenius norm of the difference
    frob_dist = np.linalg.norm(corr_gen - corr_ref, ord='fro')
    return frob_dist
