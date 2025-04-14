# 🧪 Autoregressive Diffusion Setup with Hugging Face

This guide walks you through setting up a clean environment to run autoregressive diffusion models using [lucidrains' denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) along with Hugging Face's `diffusers`, `transformers`, and `accelerate`.

---

## 🧬 Step 1: Create and Activate a Conda Environment (via Miniforge)

Before starting, make sure you have [Miniforge](https://github.com/conda-forge/miniforge) installed.

Then create and activate a new environment:

```bash
conda create -n custom_name python=3.9
conda activate custom_name
