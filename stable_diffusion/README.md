# Accelerating Stable Diffusion Inference on Intel CPUs

### How to setup benchmarking environment

```
conda create --name sb python=3.8.13 -y
conda activate sb

pip install argparse transformers diffusers accelerate torch==1.13.1
pip install intel_extension_for_pytorch==1.13.100
```
