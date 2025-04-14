import os
import torch
import numpy as np
import gc

from torch.utils.data import Dataset, DataLoader
from utils import *
from transforms import *


def compute_global_mean_std(dataloader):
    sum_layers = 0
    sum_squares = 0
    total_samples = 0

    for batch in dataloader:
        x, _ = batch  # x: (batch, layers, 1)
        x = x.squeeze(-1)  # shape: (batch, layers)

        sum_layers += x.sum(dim=0)
        sum_squares += (x ** 2).sum(dim=0)
        total_samples += x.shape[0]

    mean = sum_layers / total_samples
    std = (sum_squares / total_samples - mean ** 2).sqrt()
    
    # reshape to (1, layers, 1) for broadcasting
    return mean.view(1, -1, 1), std.view(1, -1, 1)

def check_normalization(loader, num_batches=10, batch_size=64):
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_layers = []

    for i, (layer, _) in enumerate(loader):
        all_layers.append(layer)
        if i >= num_batches - 1:
            break

    all_layers = torch.cat(all_layers, dim=0)  # shape: (N, 45, 1)

    mean_per_layer = all_layers.mean(dim=0)  # shape: (45, 1)
    std_per_layer = all_layers.std(dim=0)    # shape: (45, 1)

    print("Mean per layer (should be ~0):\n", mean_per_layer.squeeze())
    print("Std per layer (should be ~1):\n", std_per_layer.squeeze())


class NormalizedCaloChallengeDataset(Dataset):
    """ Dataset for CaloChallenge showers with normalization """
    def __init__(self, dataset, global_mean, global_std):
        """
        Arguments:
            dataset: The original dataset to wrap
            global_mean: Per-layer global mean (shape: (1, layers, 1))
            global_std: Per-layer global std (shape: (1, layers, 1))
        """
        self.dataset = dataset
        self.global_mean = global_mean
        self.global_std = global_std
        self.eps = 1e-8

    def __len__(self):
        return len(self.dataset)

    def normalize_with_global_stats(self, x):
        """ Normalize input using global mean and std """
        return (x - self.global_mean) / (self.global_std + self.eps)

    def __getitem__(self, idx):
        # Get the original data from the wrapped dataset
        layer, energy = self.dataset[idx]

        # Remove leading singleton dimension if present
        if layer.dim() == 3 and layer.shape[0] == 1:
            layer = layer.squeeze(0)  # from (1, 45, 1) to (45, 1)

        # Normalize the layer data using the global mean and std
        layer_normalized = self.normalize_with_global_stats(layer)

        #print("shape of layer_normalized: ", layer_normalized.shape)

        return layer_normalized, energy


def get_loaders(hdf5_file, particle_type, xml_filename, val_frac, batch_size,
                transforms, eps=1.e-10, device='cpu', shuffle=True, width_noise=0.0,
                single_energy=None, aug_transforms=False):

    print("Dict of preprocessing is: ")
    print(transforms)

    train_dataset = CaloChallengeDataset(hdf5_file, particle_type, xml_filename, 
                    val_frac=val_frac, transform=transforms, split='training', device=device,
                    single_energy=single_energy, aug_transforms=aug_transforms)
    
    
    val_dataset = CaloChallengeDataset(hdf5_file, particle_type, xml_filename,
                    val_frac=val_frac, transform=transforms, split='validation', device=device,
                    single_energy=single_energy, aug_transforms=aug_transforms)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    check_normalization(train_dataloader, num_batches=10, batch_size=64)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader

    
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
#     global_min, global_max = compute_global_min_max(train_dataloader)
#     print(f"Global min: {global_min}, Global max: {global_max}")
#     global_minV, global_maxV = compute_global_min_max(val_dataloader)
#     print(f"Global min val: {global_min}, Global max val: {global_max}")

#     return train_dataloader, val_dataloader, train_dataset.layer_boundaries, global_max,global_min, global_maxV, global_minV


class CaloChallengeDataset(Dataset):
    """ Dataset for CaloChallenge showers """
    def __init__(self, hdf5_file, particle_type, xml_filename, val_frac=0.3, 
            transform=None, split=None, device='cpu', single_energy=None, aug_transforms=False):
        """
        Arguments:
            hdf5_file: path to hdf5 file
            particle_type: photon, pion or electron
            xml_filename: path to XML filename
            transform: list of transformations
        """
        
        self.voxels, self.layer_boundaries = load_data(hdf5_file, particle_type, xml_filename, single_energy=single_energy)
        self.energy, self.layers = get_energy_and_sorted_layers(self.voxels)
        del self.voxels
                
        self.transform = transform
        self.aug_transforms = aug_transforms
        self.device = device
        self.dtype = torch.get_default_dtype()
        
        self.energy = torch.tensor(self.energy, dtype=self.dtype)
        self.layers = torch.tensor(self.layers, dtype=self.dtype)

        # apply preprocessing and then move to GPU
        if self.transform:
            #print("in datasets: layes and energy: ",self.layers.shape, self.energy.shape)
            for fn in self.transform:
                self.layers, self.energy = fn(self.layers, self.energy)
                
                
        # def minmax_scale(x, x_min, x_max):
        #     return 2 * (x - x_min) / (x_max - x_min) - 1
         #debugging.....
#         energy_min = self.energy.min()
#         energy_max = self.energy.max()

#         layer_min = self.layers.min()
#         layer_max = self.layers.max()
        
#         self.energy = minmax_scale(self.energy, energy_min, energy_max)
#         self.layers = minmax_scale(self.layers, layer_min, layer_max)
        
#         np.savez(
#         "scaling_stats.npz",
#         energy_min=energy_min,
#         energy_max=energy_max,
#         layer_min=layer_min,
#         layer_max=layer_max
#     )

        val_size = int(len(self.energy)*val_frac)
        trn_size = len(self.energy) - val_size
        # make train/val split
        if split == 'training':
            self.layers = self.layers[:trn_size]
            self.energy = self.energy[:trn_size]
        elif split == 'validation':
            self.layers = self.layers[-val_size:]
            self.energy = self.energy[-val_size:]
       
        self.layers = self.layers.to(device)
        self.energy = self.energy.to(device)

        # print("Dataset loaded, shape: ", self.layers.shape, self.energy.shape)
        # print(f"layers min {self.layers.min()} and layers max {self.layers.max()} , Energy min {self.energy.min()} and Energy max {self.energy.max()}")
        # print("Device: ", self.energy.device)
       

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        if self.aug_transforms:
            randint = torch.randint(high=self.layers.shape[-2], size=(1,))
            indxs = torch.arange(self.layers.shape[-2])
            rot_lay = self.layers[idx, :, :, (indxs+randint)%self.layers.shape[-2]]
            return rot_lay, self.energy[idx], self.layers[idx]
        return self.layers[idx], self.energy[idx]
