
import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import shutil
import yaml
import torch
torch.cuda.empty_cache()
from documenter import Documenter
from energyTransformer import *
from ddpm_conditional import *
from datasets import *
from transforms import *
from challenge_files import *
from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined

def get_transform_fn(name, params,doc):
    """Maps transform names to actual functions and passes parameters correctly."""
    transform_map = {
        "NormalizeByElayer": NormalizeByElayer,
        "ScaleTotalEnergy": ScaleTotalEnergy,
        "SelectDims": SelectDims,
        "ExclusiveLogitTransform": ExclusiveLogitTransform,
        "StandardizeFromFile": StandardizeFromFile,
        "LogEnergy": LogEnergy,
        "ScaleEnergy": ScaleEnergy,
        "Reshape": Reshape,
    }
    
    if name not in transform_map:
        raise ValueError(f"Unknown transform: {name}")
    if name == "StandardizeFromFile":
        model_dir = doc.basedir  # Set the correct path
        return transform_map[name](model_dir=model_dir, **params)
    # If the transformation requires parameters, pass them as keyword arguments
    return transform_map[name](**params) if params else transform_map[name]() if callable(transform_map[name]) else transform_map[name]
class prep_dataset():
    def __init__(self, params, device,doc):
        """
        :param params: file with all relevant model parameters
        """
        super().__init__()
        self.params = params
        self.device = device
        self.shape = self.params['shape']#get(self.params,'shape')
        self.trans=self.params['transforms']
        self.transform = [get_transform_fn(name, params,doc) for name, params in self.trans.items()]
        self.batch_size=self.params['batch_size']
        
        self.aug_transforms=None
        #print("self.transform:", self.transform)
        #print("Type of self.transform:", type(self.transform))
        
       

        
    def prepare_training(self):

        print("train_model: Preparing model training")

        self.train_loader, self.val_loader, self.bounds = get_loaders(
            self.params.get('hdf5_file'),
            self.params.get('particle_type'),
            self.params.get('xml_filename'),
            self.params.get('val_frac'),
            self.params.get('batch_size'),
            self.transform,
            self.params.get('eps', 1.e-10),
            device=self.device,
            shuffle=True,
            width_noise=self.params.get('width_noise', 1.e-6),
            single_energy=self.params.get('single_energy', None),
            aug_transforms=self.aug_transforms
        )
        
        self.n_trainbatches = len(self.train_loader)
        self.n_traindata = self.n_trainbatches*self.batch_size

        return self.train_loader, self.val_loader, self.bounds
        