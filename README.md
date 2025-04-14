## Environment Creation

First, git clone this GitHub repository [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/autoregressive-diffusion-pytorch). Follow their instruction to install autoregressive-diffusion-pytorch.
Next we need Diffuser, Transformer from Huggingface. Run the following command for this:
'''
pip install diffusers transformers accelerate
'''
## To run the model for DS2

Run `sbatch -o xyz.out sbatch.sh`
