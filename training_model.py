import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import shutil
import yaml
import torch
torch.cuda.empty_cache()
from documenter import Documenter
from datasets import *
from transforms import *
from challenge_files import *
from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined
from prep_data import *
import random
import datetime
import concurrent.futures
#from trainer import *
from transformer_with_diffuser import *
import itertools
import json
import time
from denoising_diffusion import *

def save_test_samples(data_loader, num_samples=5000, save_path='test_data_5k.pt', save_as_numpy=False):
    """
    Extracts num_samples from a DataLoader and saves them for fast testing.
    
    Args:
        data_loader: PyTorch DataLoader containing (x, E_inc) batches.
        num_samples: Number of samples to save.
        save_path: Path to save the extracted samples.
        save_as_numpy: If True, saves as .npz file. Otherwise, saves as .pt file.
    """
    collected_samples = []
    collected_conditions = []
    total_collected = 0

    for x_batch, E_inc_batch in data_loader:
        collected_samples.append(x_batch)
        collected_conditions.append(E_inc_batch)
        total_collected += x_batch.shape[0]
        
        if total_collected >= num_samples:
            break

    # Concatenate and trim to exactly num_samples
    x_all = torch.cat(collected_samples, dim=0)[:num_samples]
    E_inc_all = torch.cat(collected_conditions, dim=0)[:num_samples]

    if save_as_numpy:
        np.savez(save_path, x=x_all.cpu().numpy(), E_inc=E_inc_all.cpu().numpy())
        print(f"Saved {num_samples} samples to {save_path} (NumPy format).")
    else:
        torch.save({'x': x_all, 'E_inc': E_inc_all}, save_path)
        print(f"Saved {num_samples} samples to {save_path} (PyTorch format).")

    #return x_all, E_inc_all  # Optional: return for immediate use
    
def set_seed(seed=42):
    """Set seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for CuDNN
    
def main():
    set_seed(1042)
    parser = argparse.ArgumentParser(description='Fast Calorimeter Simulation')
    parser.add_argument('param_file', help='yaml parameters file')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,)
    parser.add_argument('-p', '--plot', action='store_true', default=False,)
    parser.add_argument('-d', '--saved_model_dir', default=None,)
    parser.add_argument('-ep', '--epoch', default='')
    parser.add_argument('-g', '--generate', action='store_true', default=False)
    parser.add_argument('--which_cuda', default=0) 
    parser.add_argument('-l','--loss_file_name',default='loss.png')
    parser.add_argument('-m','--model_type',default='energy')
    parser.add_argument('-t','--training_config',default='')
    parser.add_argument('-ot','--output_dir',default='')
    parser.add_argument('-mn','--model_name',default='')
    parser.add_argument('-sq','--seq_length',default=0)
    parser.add_argument('-fl','--frob_loss',default=False)
    
    args = parser.parse_args()
    print("done parsing...")
    with open(args.param_file) as f:
        yaml_params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda

    device = f'cuda:{args.which_cuda}' if use_cuda else 'cpu'
    print('device: ', device,flush=True)
    doc = Documenter(yaml_params['run_name'],base_dir=yaml_params['base_dir'] )
    # if args.model_dir:
    #     doc = Documenter(params['run_name'], base_dir=params['base_dir'],existing_run=args.model_dir)
    # else:
    #     doc = Documenter(params['run_name'],base_dir=params['base_dir'] )

    try:
        shutil.copy(args.param_file, doc.get_file('params.yaml'))
    except shutil.SameFileError:
        pass
 
    dtype = yaml_params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)

   # Load hyperparameters for training the model
    print(f"training config file: {args.training_config}")
    with open(args.training_config, 'r') as f:
        params = json.load(f)

    # Create model directory
    model_dir = f"./{args.output_dir}/{os.path.splitext(os.path.basename(args.training_config))[0]}"
    os.makedirs(model_dir, exist_ok=True)

    # Instantiate and train model
    fixed_params = {
    'max_seq_len': 45,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    # Combine with params
    all_params = {**params, **fixed_params}
  # Instantiate the prep_dataset object
    dataset_preparer = prep_dataset(params=yaml_params, device=device,doc=doc)
    #Call the prepare_training function
    train_loader, val_loader,transform=dataset_preparer.prepare_training()
    if args.generate==False:
      
        if args.model_name=='DDM':
            model = DenoisingDiffusion(**all_params) #match all the params
            optimizer = torch.optim.Adam(model.parameters(), yaml_params['lr']) 
            # Run training
            fit(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=all_params['device'],
                epochs=yaml_params['epochs'],
                model_dir=model_dir,
                frob_loss=args.frob_loss,
                seq_length=args.seq_length
            )
        elif args.model_name=='ADM':
            model = AutoregressiveDiffusion(**all_params) #match all the params
            optimizer = torch.optim.Adam(model.parameters(), yaml_params['lr'])

            # Run training
            fit(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=all_params['device'],
                epochs=yaml_params['epochs'],
                model_dir=model_dir,
                frob_loss=args.frob_loss,
                seq_length=args.seq_length
            )
        else:
            print("Not implemented yet!")
    else:
        print("inside the sampling condition: ",yaml_params.get('particle_type'))
        # Dataset & Dataloader
        test_dataset = CaloChallengeDataset(
            yaml_params.get('eval_hdf5_file'),
            yaml_params.get('particle_type'),
            yaml_params.get('xml_filename'),
            transform=transform,
            device=device,
            single_energy=None
        )

        batch_size_sample = yaml_params.get('batch_size_sample')
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_sample, shuffle=False)

        # Model Loading (assuming state_dict)
        model_path = args.saved_model_dir
        if args.model_name=='DDM':
            model = DenoisingDiffusion(**all_params) #match all the params
        elif args.model_name=='ADM':
            model = AutoregressiveDiffusion(**all_params)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        epoch=-1
        #num_samples can be loaded from the config file...
        mse,sampling_time=model.sampling_val_loop(test_dataloader,device,epoch='test',model_dir=model_path,n_samples=yaml_params['n_samples'])
        print(f"sampling mse: {mse} and sampling time: {sampling_time} ")


  

    #print(f"Finished experiment: {config_path}")
if __name__ == "__main__":
    main()