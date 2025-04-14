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
from trainer import *
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
    set_seed(1042)
    parser = argparse.ArgumentParser(description='Fast Calorimeter Simulation')
    parser.add_argument('param_file', help='yaml parameters file')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,)
    parser.add_argument('-p', '--plot', action='store_true', default=False,)
    parser.add_argument('-d', '--model_dir', default=None,)
    parser.add_argument('-ep', '--epoch', default='')
    parser.add_argument('-g', '--generate', action='store_true', default=False)
    parser.add_argument('--which_cuda', default=0) 
    parser.add_argument('-l','--loss_file_name',default='loss.png')

    args = parser.parse_args()
    print(args.param_file)

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda

    device = f'cuda:{args.which_cuda}' if use_cuda else 'cpu'
    print('device: ', device,flush=True)

    if args.model_dir:
        doc = Documenter(params['run_name'], base_dir=params['base_dir'],existing_run=args.model_dir)
    else:
        doc = Documenter(params['run_name'],base_dir=params['base_dir'] )

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
    train_loader, val_loader=dataset_preparer.prepare_training()
    data=[train_loader, val_loader]
    
    
    model = AutoregressiveDiffusion(
    dim_input = 1,
    dim = 16,
    max_seq_len = 45,
    depth = 2 )

    trainer = ModelTrainer(
    model = model,
        learning_rate=params.get('lr',5.e-3),
        num_train_steps=params.get('epochs',5),
    train_dataloader= data[0],
        val_dataloader=data[1],args=args
    )
    print("before calling trainer....",flush=True)
    trainer()
    
    trainer.sampling_layers(params=params,
    dataset_preparer=dataset_preparer,
    model=model,
    args=args,
    device=trainer.accelerator.device,  # or just torch.device('cuda') / 'cpu'
    doc=doc  # or None if you're not using a documenter
)

if __name__=='__main__':
    main()
  
