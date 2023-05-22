# How to set envoriment

```
conda create --name opt python=3.9 -y
conda activate opt
pip install argparse transformers accelerate torch==2.0.0
pip install intel_extension_for_pytorch
pip install Xformers
```