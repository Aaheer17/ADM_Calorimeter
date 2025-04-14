from typing import List

import math
from pathlib import Path
import time
from accelerate import Accelerator
from ema_pytorch import EMA
import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import shutil
import yaml
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from torchvision.utils import save_image
import torchvision.transforms as T
torch.cuda.empty_cache()
from autoregressive_unet import AutoregressiveDiffusion
from documenter import Documenter
from datasets import *
from transforms import *
from prep_data import *
import random
from utils import *
from PIL import Image
import matplotlib.pyplot as plt

   
class ModelTrainer(Module):
    def __init__(
        self,
        model,
        *,
        train_dataloader,
        val_dataloader,
        args,
        num_train_steps = 1000,
        learning_rate = 1e-5,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.model = model

        self.optimizer = Adam(model.parameters(), lr = learning_rate, **adam_kwargs)
        
        self.args=args
        
        self.dl = train_dataloader  
        self.val_dataloader=val_dataloader
        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

       

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    

    def forward(self):

        #dl = cycle(self.dl)
        num_epochs=self.num_train_steps
        epoch_loss=[]
        val_losses=[]
        val_mean=[]
        val_mae=[]
        MSE=[]
        max_grad_norm = 1.0  # Set a threshold for gradient clipping
        print("STARTING>>>>")
        for epoch in range(num_epochs):  # Define the number of epochs
            print(f"Epoch {epoch+1}/{num_epochs}",flush=True)
            self.model.train()
            avg_loss=0
            for step, data in enumerate(self.dl):  # Iterate over dataloader properly
                global_step = epoch * len(self.dl) + step + 1  # Track overall steps
                
                loss = self.model(data)
                avg_loss+=loss.item()

                #self.accelerator.print(f'[{global_step}] loss: {loss.item():.3f}')
                self.accelerator.backward(loss)
                

                # Apply gradient clipping **before** optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.accelerator.unwrap_model(self.model).parameters(), max_grad_norm)
               

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.accelerator.wait_for_everyone()


            
            epoch_loss.append(avg_loss/(step+1))
            print(f"loss calculated {avg_loss/(step+1)}")
        
            val_loss = self.validate()
            if val_loss is not None:
                val_losses.append(val_loss)
                
                
            mean_diff, mae_layerwise,mse=self.sampling_validate(num_samples=2048)
            val_mean.append(mean_diff)
            val_mae.append(mae_layerwise)
            MSE.append(mse)
            
        
        plot_losses(epoch_loss, val_losses, title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss",file_name=self.args.loss_file_name)
   

        print('Training complete and printing avg training losses! ',epoch_loss)
    def validate(self):
        """Runs validation on the validation dataset."""
        if self.val_dataloader is None:
            return  # Skip validation if no validation dataloader is provided

        self.model.eval()  # Set model to evaluation mode
        val_loss = 0
        num_batches = 0

        with torch.no_grad():  # Disable gradient computation for efficiency
            for data in self.val_dataloader:
                loss = self.model(data)
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches
        print(f'Validation Loss: {avg_val_loss:.3f}')
        return avg_val_loss
    
    def sampling_validate(self, num_samples=100):
        """
        Sampling-based validation. Generates samples using the model and compares them
        to real validation data using custom physics/statistical metrics.
        """
        if self.val_dataloader is None:
            return

        self.model.eval()
        generated = []
        reference = []
        energies = []

        with torch.no_grad():
            sample_count = 0
            for batch in self.val_dataloader:
                # Adjust this depending on your dataloader structure
                real_shower, Einc = batch  # (B, 45, 1), (B, 1)
                batch_size = real_shower.shape[0]

                # Generate synthetic showers
                gen_shower = self.model.sample(batch_size=batch_size, incident_energy=Einc)

                # Store for metric comparison
                generated.append(gen_shower.cpu())
                reference.append(real_shower.cpu())
                energies.append(Einc.cpu())

                sample_count += batch_size
                if sample_count >= num_samples:
                    break

        # Stack all outputs
        
        generated = torch.cat(generated, dim=0)[:num_samples]  # (N, 45, 1)
        generated=generated[:,1:]
        reference = torch.cat(reference, dim=0)[:num_samples]  # (N, 45, 1)
        energies = torch.cat(energies, dim=0)[:num_samples]    # (N, 1)

        print("Generated samples shape:", generated.shape)
        print("Reference samples shape:", reference.shape)

        # Example validation metrics
        gen_total_energy = generated.sum(dim=1).squeeze()
        ref_total_energy = reference.sum(dim=1).squeeze()

        mean_diff = (gen_total_energy - ref_total_energy).abs().mean().item()
        print(f"Mean total energy difference: {mean_diff:.4f}")

        # Optional: layer-wise mean/std
        gen_layer_mean = generated.mean(dim=0).squeeze()
        ref_layer_mean = reference.mean(dim=0).squeeze()
        mae_layerwise = (gen_layer_mean - ref_layer_mean).abs().mean().item()
        print(f"Mean absolute error per layer: {mae_layerwise:.4f}")
        
        mse = F.mse_loss(generated, reference).item()
        print(f"Mean Squared Error (per voxel): {mse:.4f}")

        return mean_diff, mae_layerwise,mse

        # Add more: CFD, correlations, histogram matching etc. as needed

    
    def sampling_layers(self,params,dataset_preparer,model,args,device, doc):
        
        #initialize random incident energy
        Einc = torch.tensor(
            10**np.random.uniform(3, 6, size = params.get("n_samples", 10**5)) ,    
            dtype=torch.get_default_dtype(),
            device=device
        ).unsqueeze(1)
        
        # transform Einc to basis used in training
        dummy, transformed_cond = None, Einc
        transform=dataset_preparer.transform
        for fn in transform:
            print("fn: ",fn)
            if hasattr(fn, 'cond_transform'):
               
                dummy, transformed_cond = fn(dummy, transformed_cond)
                
        batch_size = params.get("batch_size_sample", 10000)
            
        transformed_cond_loader = DataLoader(dataset=transformed_cond, batch_size=batch_size, shuffle=False)
            
        all_generated = []
        start_time = time.time()  # ⏱️ start timing

        for batch in transformed_cond_loader:
            incident_energy = batch.to(model.device)  # move to GPU/CPU
            batch_size = incident_energy.shape[0]

            # Call the model's sampling method
            generated = model.sample(batch_size=batch_size, incident_energy=incident_energy)

            # generated: shape (batch_size, seq_len, 1)
            all_generated.append(generated.cpu())  # store or save later
        end_time = time.time()  # ⏱️ end timing
        

        # Concatenate all generated batches
        all_generated = torch.cat(all_generated, dim=0)  # shape: [N, 45, 1]
        elapsed_time = end_time - start_time
        print(f"Sampling completed in {elapsed_time:.2f} seconds.")  
        total_samples = all_generated.shape[0]

        time_per_sample = elapsed_time / total_samples
        print(f"Samples generated: {total_samples}")
        print(f"Per-sample generation time: {time_per_sample:.6f} seconds")

        samples = all_generated.squeeze(-1)[:,1:]
        torch.save(samples, self.results_folder / "samples.pt")
        ref_dataset = CaloChallengeDataset(
                    params.get('eval_hdf5_file'),
                    params.get('particle_type'),
                    params.get('xml_filename'),
                    transform=transform, # TODO: Or, apply NormalizeEByLayer popped from model transforms
                    device=device,
                    single_energy=None
                )
        reference=ref_dataset.layers
        ref_energy=ref_dataset.energy
        #going through the reverse of previously applied transformation
        #except for the NormalizeBYELayer
        for fn in transform[::-1]:
            if fn.__class__.__name__ == 'NormalizeByElayer':
                break # this might break plotting
            samples, _ = fn(samples, Einc, rev=True) 
            reference, _ = fn(reference, ref_energy, rev=True) 

        # # clip u_i's (except u_0) to [0,1] 
        samples[:,1:] = torch.clip(samples[:,1:], min=0., max=1.)
        reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)
        
        plot_mean_and_std_samples(samples,reference,self.results_folder,name='mean_std.pdf')

            
    
        #plt.show()




        
        
        
        
        

        
        
        
        

