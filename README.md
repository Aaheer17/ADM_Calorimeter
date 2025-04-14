# 🧪 Autoregressive Diffusion Setup with Hugging Face

This guide walks you through setting up a clean environment to run autoregressive diffusion models using [lucidrains' autoregressive-diffusion-pytorch](https://github.com/lucidrains/autoregressive-diffusion-pytorch) along with Hugging Face's `diffusers`, `transformers`, and `accelerate`.

---
## 🧬 Step 1: Git clone this repository
```bash
git clone -b layer_model_v1 --single-branch https://github.com/Aaheer17/ADM_Calorimeter.git
```
## 🧬 Step 2: Create and Activate a Conda Environment (via Miniforge)

Before starting, make sure you have [Miniforge](https://github.com/conda-forge/miniforge) installed.

Then create and activate a new environment:

```bash
conda create -n custom_name python=3.9
conda activate custom_name

git clone https://github.com/lucidrains/autoregressive-diffusion-pytorch.git
cd autoregressive-diffusion-pytorch
pip install autoregressive-diffusion-pytorch
pip install diffusers transformers accelerate
```
# 🧪 Running the Model

To train the layer model and generate samples, follow this command

```bash
sbatch submit.sh
