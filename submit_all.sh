#!/bin/bash

# Loop over config numbers (adjust as needed)
for i in {0..2}
do
  sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=config_${i}
#SBATCH -t 12:00:00
#SBATCH --mem=80000
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:1
#SBATCH -A bii_nssac
#SBATCH --output=training_config_${i}.out

module load miniforge
source activate ardiff

# To train the model,uncomment this block
#python3 training_model.py \
#  /project/biocomplexity/fa7sa/Diffusion_multi_step/config.yaml \
#  --use_cuda \
#  -m 'energy' \
#  --training_config "./test_config/config_${i}.json" \
#  --output_dir "test_results" \
#  --model_name "DDM"

# To generate sample, uncomment this block
python3 training_model.py \
   /project/biocomplexity/fa7sa/Diffusion_multi_step/config.yaml \
   --use_cuda \
   -m 'energy' \
   -d "./test_results/config_${i}/best_val_loss_model.pt" \
   -g \
   --training_config "./test_config/config_${i}.json"\
   --output_dir "test_results" \
   --model_name "DDM"

EOT
done
