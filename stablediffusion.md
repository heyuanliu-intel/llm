# Large Language Model

### How to run CPU Pipeline

```
git clone https://github.com/CompVis/stable-diffusion.git
cd stable-diffusion/
conda env create -f environment.yaml
conda activate ldm
pip install accelerate
```

Edit the environment.yaml and downgrade diffusers version:

```
diffusers==0.12.1
transformers==4.24.0
```

Then execute below command to update env.

```
conda env update -f environment.yaml
```

Download sdv1.4 models:

```
cd ~
curl https://f004.backblazeb2.com/file/aai-blog-files/sd-v1-4.ckpt > sd-v1-4.ckpt
cd stable-diffusion/
mkdir -p models/ldm/stable-diffusion-v1/
ln -s ~/sd-v1-4.ckpt ~/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
```

### Errors and how to fix:

1. Errors: ImportError: cannot import name 'SAFE_WEIGHTS_NAME' from 'transformers.utils'

Edit the environment.yaml

```
diffusers==0.12.1
```

Then execute below command to update env.

```
conda env update -f environment.yaml
```

2.AttributeError: module transformers has no attribute CLIPImageProcessor

Edit the environment.yaml

```
transformers==4.27.4
```

Then execute below command to update env.

```
conda env update -f environment.yaml
```

3. RuntimeError: expected scalar type Float but found BFloat16

Torch version issue.

```
 conda create --name llm python=3.8.13
 pip install torch==1.13.1
```
