# Accelerating Stable Diffusion Inference on Intel CPUs

### How to setup benchmarking environment for Baseline Benchmarking and IPEX

```
conda create --name sb python=3.8.13 -y
conda activate sb

pip install argparse transformers diffusers accelerate torch==1.13.1
pip install intel_extension_for_pytorch==1.13.100

cd ~/llm
./benchmark_sb.sh
```

### How to setup benchmarking environment for OpenVINO

```
conda create --name sb_ov python=3.8.13 -y
conda activate sb_ov

pip install optimum[openvino]
cd ~/llm
./bench_sb_ov.sh
```
