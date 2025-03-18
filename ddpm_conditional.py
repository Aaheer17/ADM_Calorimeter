import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext
from datasets import *
from documenter import Documenter
from plotting_util import *
from transforms import *
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import *
from modules import UNet_conditional, EMA,EarlyStopping
from energyTransformer import *
import os, time
from prep_data import *
from challenge_files import *
from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined
from einops import rearrange
import csv

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=400, beta_start=1e-4, beta_end=0.02, num_layers=45, device="cuda", params=None,doc=None):
        
        self.params=params
        self.noise_steps = 100
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        ## transformation data
        
        self.trans=self.params['transforms']
        self.transforms = [get_transform_fn(name, params,doc) for name, params in self.trans.items()]
    
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.num_layers = num_layers
        self.emb_dim=params.get('emb_dim', 64)
        self.model = UNet_conditional(45, 45, self.emb_dim,dropout=0.2, use_bn=True).to(device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(" ######### total num of parameters: #######", total_params)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        
        self.device = device
        self.epochs=self.params.get('epochs',100)
        self.lr=self.params.get('lr',1.e-4)
        self.max_lr=self.params.get('max_lr',5.e-4)
        self.do_validation=self.params.get('do_validation',True)
        self.etransformer = energyTransformer(self.params).to(device)
        
        self.t_min = get(self.params, "t_min", 0)
        self.t_max = get(self.params, "t_max", 1)
        distribution = get(self.params, "distribution", "uniform")
        if distribution == "uniform":
            self.distribution = torch.distributions.uniform.Uniform(low=self.t_min, high=self.t_max)
        elif distribution == "beta":
            self.distribution = torch.distributions.beta.Beta(1.5, 1.5)
        else:
            raise NotImplementedError(f"build_model: Distribution type {distribution} not implemented")

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_autoregressive(self, E_inc, batch_size):
        print("Starting autoregressive sampling...")
        self.model.load_state_dict(torch.load(os.path.join(self.doc.basedir, f"ckpt.pt"), map_location=E_inc.device))
        self.model.eval()
        dtype = E_inc.dtype
        device = E_inc.device
        #print("self.noise_steps: ",self.noise_steps)
        timesteps = torch.linspace(self.noise_steps - 1, 0, self.noise_steps, device=device, dtype=dtype)
        with torch.no_grad():
            embedding=self.etransformer(E_inc, None, None,True)
            embedding = F.pad(embedding, (0, 0, 0, self.num_layers - embedding.shape[1]), "constant", 0)
            if torch.isnan(embedding).any():
                print("ALERT: embedding has nan before starting the loop!")
            
            for layer in range(self.num_layers):
                x_t = torch.randn((batch_size, self.num_layers, 1), device=device, dtype=dtype)
                for t in tqdm(timesteps, desc=f"Sampling Layer {layer + 1}", leave=True):
                    t_torch = torch.full((batch_size, self.num_layers, 1),
                                         t / self.noise_steps * 2 - 1, device=device, dtype=dtype)
                    print("before time embedding: ",t_torch.shape)
                    t_embed=self.etransformer.time_embedding(t_torch) ##is this causing an issue?
                    if torch.isnan(t_embed).any():
                        print("ALERT: time embedding has nan !")
                    print(f"x_t nan: {torch.isnan(x_t).any()} and embedding nan: {torch.isnan(embedding).any()}")
                    predicted_noise=self.model(x_t, t_embed, embedding)
                    if torch.isnan(predicted_noise).any():
                        print("ALERT: predicted noise has nan!")
                    #check if this transformation is correct........
                    alpha_t = self.alpha_hat[int(t)]
                    beta_t = self.beta[int(t)]
                    sigma_t = torch.sqrt(beta_t)

                    if t > 0:
                        z = torch.randn_like(x_t)
                    else:
                        z = 0

                    x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_t)) * predicted_noise) + sigma_t * z
                    if torch.isnan(x_t).any():
                        print("ALERT: after modification x_t has nan!")
                print(f"Done with layer {layer}")
                if layer<=43:
                    #print("shape of x_t: ",x_t[:,:layer+1].shape)
                    embedding=self.etransformer(E_inc, None, x_t[:,:layer+1],rev=True)
                    #print("before finishing embedding shape: ",embedding.shape)
                    embedding = F.pad(embedding, (0, 0, 0, self.num_layers - embedding.shape[1]), "constant", 0)
                    print(f"Finished layer{layer} and embedding shape is: ",embedding.shape)
                    
                    if torch.isnan(embedding).any():
                        print("ALERT: embedding has nan inside the loop!")
            
            
                    
        print("Autoregressive sampling complete. Output shape:", x_t.shape)
        if torch.isnan(x_t).any():
            print("ALERT:x_t has nan before returning!")
        return x_t            
                    
                

        
    
    def noise_samples(self, layer_energy, t):
        #print(f"t: {t}, dtype: {t.dtype}, shape: {t.shape}")  # Debugging

        # Normalize `t` to [0, T-1] and interpolate `alpha_hat`
        t_scaled = t * (self.noise_steps - 1)  # Scale `t` from [0,1] to [0,999]

        # Ensure valid indexing (avoid out-of-bounds errors)
        t_floor = torch.floor(t_scaled).long().clamp(0, self.noise_steps - 2)  # (batch_size, 45, 1)
        t_ceil = (t_floor + 1).clamp(0, self.noise_steps - 1)
        w = t_scaled - t_floor  # Interpolation weight

        # Interpolate `alpha_hat`
        alpha_floor = self.alpha_hat[t_floor]  # (batch_size, 45, 1)
        alpha_ceil = self.alpha_hat[t_ceil]  # (batch_size, 45, 1)
        alpha_interp = (1 - w) * alpha_floor + w * alpha_ceil  # Weighted sum

        # Compute noise factors (no extra dimensions)
        sqrt_alpha_hat = torch.sqrt(alpha_interp)  
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_interp)  

        # Sample noise
        noise = torch.randn_like(layer_energy)
        #print("shape of noise: ",noise.shape)
        return sqrt_alpha_hat * layer_energy + sqrt_one_minus_alpha_hat * noise, noise

    def train_step(self, loss):
        """Performs one training step with gradient scaling and EMA updates."""
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Check if gradients are updating in model
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                logging.info(f"Grad Update (Model) - {name}: {param.grad.abs().mean().item()}")

        # Check if gradients are updating in etransformer
        for name, param in self.etransformer.named_parameters():
            if param.requires_grad and param.grad is not None:
                logging.info(f"Grad Update (Transformer) - {name}: {param.grad.abs().mean().item()}")

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()


    def one_epoch(self, train=True):
        avg_loss = 0.
        loader = self.train_loader if train else self.val_loader
        if train:
            self.model.train()
            self.etransformer.train()
        else:
            self.model.eval()
            self.etransformer.eval()

        b_counter = 0
        with torch.no_grad() if not train else nullcontext():
            for batch_id, x in enumerate(loader):
                b_counter += 1
                layer_energy, E_inc = x[0].to(self.device), x[1].to(self.device).unsqueeze(-1)

                # Sample timesteps
                t = self.distribution.sample(
                    list(layer_energy.shape[:2]) + [1] * (layer_energy.ndim - 2)
                ).to(dtype=layer_energy.dtype, device=layer_energy.device)

                # Generate noisy samples
                x_t, noise = self.noise_samples(layer_energy, t)

                # Get transformer embeddings
                embedding, t_embed = self.etransformer(E_inc, t, layer_energy, rev=False)

                # Predict noise
                predicted_noise = self.model(x_t, t_embed, embedding)

                # Check for NaN issues
                if torch.isnan(predicted_noise).any():
                    logging.warning("ALERT: Predicted noise contains NaNs!")
                    continue

                # Compute loss
                loss = self.mse(noise, predicted_noise)
                if torch.isnan(loss).any():
                    logging.warning("ALERT: Loss contains NaNs!")
                    continue  # Skip this batch to avoid instability

                avg_loss += loss.item()

                if train:
                    self.train_step(loss)

                    # Check if model weights are updating
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            weight_change = param.abs().mean().item()
                            logging.info(f"Weight Update (Model) - {name}: {weight_change}")

                    # Check if etransformer weights are updating
                    for name, param in self.etransformer.named_parameters():
                        if param.requires_grad:
                            weight_change = param.abs().mean().item()
                            logging.info(f"Weight Update (Transformer) - {name}: {weight_change}")
                        if param.requires_grad:
                            logging.info(f"Param: {name}, Grad: {param.grad.abs().mean().item()}")

                    
                        
                    # Log learning rate and loss
                    logging.info(f"Batch {batch_id} | Loss: {loss.item():.6f} | LR: {self.scheduler.get_last_lr()[0]}")

        print(f"Processed Batches: {b_counter}/{len(loader)}")
        avg_loss /= max(b_counter, 1)  # Prevent division by zero
        return avg_loss


    def prepare(self, data):
        self.train_loader, self.val_loader = data[0],data[1]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=1e-5, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.max_lr, 
                                                 steps_per_epoch=len(self.train_loader), epochs=self.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()
    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
    
    def sample_layer(self,n_samples=10**5):
        t_0 = time.time()
        #if self.params['eval_dataset'] in ['2', '3']
        Einc = torch.tensor(
            10**np.random.uniform(3, 6, size=get(self.params, "n_samples", 10**5)) ,    
            dtype=torch.get_default_dtype(),
            device=self.device
        )
        Einc=rearrange(Einc,"a->a 1 1")
        
        dummy, transformed_cond = None, Einc
        #print("starting transforming for sampling")
        for fn in self.transforms:
            
            if hasattr(fn, 'cond_transform'):
                dummy, transformed_cond = fn(dummy, transformed_cond)

        batch_size_sample = get(self.params, "batch_size_sample", 1000)
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
        )
        
        sample = torch.vstack([self.sample_autoregressive(c,batch_size_sample).cpu() for c in transformed_cond_loader])
        
        
        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sample_time"] = sampling_time
        print(
            f"generate_samples: Finished generating {len(sample)} samples "
            f"after {sampling_time} s.", flush=True
        )

        return sample, transformed_cond.cpu()
    
    
    
    def plot_samples(self, samples, conditions, name="", energy=None, doc=None):
        
        transforms = self.transforms
        if doc is None: doc = self.doc
        self.single_energy=energy
        if self.params['model_type'] == 'energy':
            reference = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=transforms, # TODO: Or, apply NormalizeEByLayer popped from model transforms
                device=self.device,
                single_energy=self.single_energy
            ).layers
            
            # postprocess
            for fn in transforms[::-1]:
                if fn.__class__.__name__ == 'NormalizeByElayer':
                    break # this might break plotting
                samples, _ = fn(samples, conditions, rev=True)
                reference, _ = fn(reference, conditions, rev=True)

            # clip u_i's (except u_0) to [0,1] 
            samples[:,1:] = torch.clip(samples[:,1:], min=0., max=1.)
            reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)
            print("before ploting ui_dists")
            plot_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc
            )
            print("before eval ui dists")
            evaluate.eval_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
                params=self.params,
            )
        else:
            if self.latent:
                #save generated latent space
                self.save_sample(samples, conditions, name=name+'_latent', doc=doc) 
                    
            # postprocess
            for fn in transforms[::-1]:
                samples, conditions = fn(samples, conditions, rev=True)
            
            samples = samples.detach().cpu().numpy()
            conditions = conditions.detach().cpu().numpy()

            self.save_sample(samples, conditions, name=name, doc=doc)
            evaluate.run_from_py(samples, conditions, doc, self.params)

    def plot_saved_samples(self, name="", energy=None, doc=None):
        if doc is None: doc = self.doc
        mode = self.params.get('eval_mode', 'all')
        script_args = (
            f"-i {doc.basedir}/ "
            f"-r {self.params['eval_hdf5_file']} -m {mode} --cut {self.params['eval_cut']} "
            f"-d {self.params['eval_dataset']} --output_dir {doc.basedir}/final/ --save_mem"
        ) + (f" --energy {energy}" if energy is not None else '')
        evaluate.main(script_args.split())

    def save_sample(self, sample, energies, name="", doc=None):
        """Save sample in the correct format"""
        if doc is None: doc = self.doc
        save_file = h5py.File(doc.get_file(f'samples{name}.hdf5'), 'w')
        save_file.create_dataset('incident_energies', data=energies)
        save_file.create_dataset('showers', data=sample)
        save_file.close()            
        
    def fit(self, doc):
        self.doc=doc
        best_val_loss = float('inf')  # Initialize with a high value
        
        total_start_time = time.time()  # Track total training time
        train_losses=[]
        val_losses=[]
        self.early_stopping = EarlyStopping(patience=5, delta=0.01, path="best_model.pth",doc=self.doc)
        for epoch in tqdm(range(self.epochs), desc="Training Progress", leave=True):
            
            logging.info(f"Starting epoch {epoch}:")
            epoch_start_time = time.time()  # Track time per epoch
            avg_loss_train = self.one_epoch(train=True)
            train_losses.append(avg_loss_train)
            epoch_train_time = time.time() - epoch_start_time  # Calculate training time per epoch
            print(f"Epoch {epoch} Training Time: {epoch_train_time:.2f} sec")
            logging.info(f"Epoch {epoch} Training Time: {epoch_train_time:.2f} sec")
            logging.info(f"Avg loss after training: {avg_loss_train}")
            print("Validation: ",self.do_validation)
            if self.do_validation:
                
                avg_loss_val = self.one_epoch(train=False)
                val_losses.append(avg_loss_val)
                logging.info(f"Validation MSE: {avg_loss_val}")
                if avg_loss_val < best_val_loss:
                    best_val_loss = avg_loss_val  # Update best loss
                    logging.info(f"New best validation loss: {best_val_loss}. Saving model...")
                    self.save_model(self.doc.basedir, epoch=epoch)
                self.early_stopping(avg_loss_val, self.model,self.doc)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break
        total_training_time = time.time() - total_start_time  # Calculate total training time
        logging.info(f"Training completed. Best validation loss: {best_val_loss}")
        # Define CSV file name
        csv_filename = "loss_values.csv"

        # Write to CSV
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Write header
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

            # Write loss values
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
                writer.writerow([epoch, train_loss, val_loss])

        plot_losses(train_losses, val_losses, filename='loss_plot.png',doc=doc)
        if get(self.params, "sample", True):
            print("generate_samples: Start generating samples", flush=True)
            n_samples=get(self.params,'n_samples',1000)
            samples,c=self.sample_layer(n_samples)
            self.plot_samples(samples=samples, conditions=c)
            ##self.plot_samples(samples=samples, conditions=c, energy=self.single_energy)

if __name__ == '__main__':
    parse_args(config)
    set_seed(config.seed)
    diffuser = Diffusion(config.noise_steps, num_layers=config.num_layers)
    diffuser.prepare(config)
    diffuser.fit(config)
