## creating the config file.....
import itertools
import torch
import json
import os
#current file will generate 91 configurations..
#commented out params are for transformer
param_grid = {
    'block_out_channels': [(4, 8, 16, 16), (4, 4, 8, 8)],
    'cross_attention_dim': [8, 16],
    #'trans_dim': [32],
    'layers_per_block': [2],
    'norm_num_groups': [4],
    #'heads': [2],
    #'num_encoder_layers': [2],
    #'num_decoder_layers': [2],
    #'dropout_p': [0.05, 0.1],
    'diffusion_timestep': [100, 200],
    'schedule_type': ['DDPM','DDIM'],
    'num_inference_steps':[50,100],  # should be less than diffusion_step
    'prediction_type':['sample','epsilon','v-prediction']
    
}
# Generate all possible combinations
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


#DD stands for denoising diffusion
for idx, params in enumerate(param_combinations):
    config_dir = './configs_DD'
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f'config_{idx}.json')

    with open(config_path, 'w') as f:
        json.dump(params, f, indent=4)
        
        
        
"""
To submit jobs:
import glob
import os

config_files = glob.glob('./PATH_TO_CONFIG_FOLDER/*.json')

for cfg in config_files:
    os.system(f"sbatch submit.sh {cfg}")

"""
