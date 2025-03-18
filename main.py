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
from prep_data import *
import random
#import wandb

def set_seed(seed=42):
    """Set seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for CuDNN



def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Fast Calorimeter Simulation')
    parser.add_argument('param_file', help='yaml parameters file')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,)
    parser.add_argument('-p', '--plot', action='store_true', default=False,)
    parser.add_argument('-d', '--model_dir', default=None,)
    parser.add_argument('-ep', '--epoch', default='')
    parser.add_argument('-g', '--generate', action='store_true', default=False)
    parser.add_argument('--which_cuda', default=0) 

    args = parser.parse_args()
    print(args.param_file)

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda

    device = f'cuda:{args.which_cuda}' if use_cuda else 'cpu'
    #print('device: ', device)

    if args.model_dir:
        doc = Documenter(params['run_name'], existing_run=args.model_dir)
    else:
        doc = Documenter(params['run_name'])

    try:
        shutil.copy(args.param_file, doc.get_file('params.yaml'))
    except shutil.SameFileError:
        pass
 
    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)


    # Instantiate the prep_dataset object
    dataset_preparer = prep_dataset(params=params, device=device,doc=doc)

    # Call the prepare_training function
    train_loader, val_loader,bound=dataset_preparer.prepare_training()
    data=[train_loader, val_loader,bound]
    diffuser = Diffusion(noise_steps=1000, num_layers=45,device=device, params=params,doc=doc)
    # with wandb.init(project="train_calorimeter", group="train", config=config):
    diffuser.prepare(data)
    diffuser.fit(doc)
    
    # save parameter file with new entries
    # with open(doc.get_file('final_params.yaml'), 'w') as f:
    #     yaml.dump(model.params, f, default_flow_style=False)

if __name__=='__main__':
    main()
  