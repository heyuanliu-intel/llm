
# GPU

sudo apt-get update
yes | sudo apt-get install python3.8
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
bash Miniconda3-py38_4.12.0-Linux-x86_64.sh

How to update conda:
conda update -n base -c defaults conda --repodata-fn=repodata.json

git clone https://github.com/CompVis/stable-diffusion.git
cd stable-diffusion/
conda env create -f environment.yaml
conda activate ldm

curl https://f004.backblazeb2.com/file/aai-blog-files/sd-v1-4.ckpt > sd-v1-4.ckpt

mkdir -p models/ldm/stable-diffusion-v1/
ln -s ~/sd-v1-4.ckpt ~/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt

python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 

python scripts/txt2img.py --prompt "a photorealistic vaporwave image of a lizard riding a snowboard through space" --plms --ckpt sd-v1-4.ckpt --skip_grid --n_samples 1

# CPU
sudo apt-get update
yes | sudo apt-get install python3.8

git clone https://github.com/bes-dev/stable_diffusion.openvino.git
cd stable_diffusion.openvino

python -m pip install --upgrade pip
pip install openvino-dev[onnx,pytorch]==2022.3.0
pip install -r requirements.txt

pip install -r requirements.txt

mkdir -p models/ldm/stable-diffusion-v1/
ln -s ~/sd-v1-4.ckpt ~/stable_diffusion.openvino/models/ldm/stable-diffusion-v1/model.ckpt

Example Text-To-Image
python demo.py --prompt "bright beautiful solarpunk landscape, photorealism"