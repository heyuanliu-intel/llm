# Accelerating Stable Diffusion Inference on Intel CPUs

### How to setup benchmarking environment for Baseline Benchmarking and IPEX (Torch 1.13.1)

```
conda create --name sb python=3.8.13 -y
conda activate sb

pip install argparse transformers diffusers accelerate torch==1.13.1
pip install intel_extension_for_pytorch==1.13.100

cd ~/llm
./benchmark_sb.sh
```

### How to setup benchmarking environment for OpenVINO (Torch 2.0)

```
conda create --name sb_ov python=3.8.13 -y
conda activate sb_ov

pip install optimum[openvino]
pip install argparse transformers diffusers accelerate intel_extension_for_pytorch
cd ~/llm
./bench_sb_ov.sh
```

### System-level optimization

```
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
yum install -y intel-mkl --nogpgcheck
conda install -c conda-forge jemalloc

```
