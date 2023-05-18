import time
import argparse
import intel_extension_for_pytorch as ipex
import torch
from diffusers import StableDiffusionPipeline

# model_id = "runwayml/stable-diffusion-v1-5"
# Use cached folder to avoid download the models from network
model_id = "/home/heyuan/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a/"
prompt = "sailing ship in storm by Rembrandt"
nb_pass = 10


def elapsed_time(pipeline, nb_pass=10, num_inference_steps=20):
    start = time.time()
    for _ in range(nb_pass):
        _ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np")
    end = time.time()
    return (end - start) / nb_pass


def bench_float32():
    # build a StableDiffusionPipeline with the default float32 data type
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")
    # warmup
    images = pipe(prompt, num_inference_steps=10).images
    # benchmarking
    latency = elapsed_time(pipe, nb_pass=nb_pass)
    print(f"benchmark standard pipeline with float32 and latency is {latency}")


def bench_ipex_bf16():
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    # to channels last
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
    pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)

    sample = torch.randn(2, 4, 64, 64)
    timestep = torch.rand(1)*999
    encoder_hidden_status = torch.randn(2, 77, 768)
    input_example = (sample, timestep, encoder_hidden_status)

    # optimize with ipex
    pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True, sample_input=input_example)
    pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=torch.bfloat16, inplace=True)
    pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
    pipe.safety_checker = ipex.optimize(pipe.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)

    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        latency = elapsed_time(pipe, nb_pass=nb_pass)
        print(f"benchmark standard pipeline with ipex and bf16 and latency is {latency}")


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--baseline', action='store_true', help='A boolean switch')
parser.add_argument('--ipex', action='store_true', help='A boolean switch')
args = parser.parse_args()

if args.baseline:
    bench_float32()
elif args.ipex:
    bench_ipex_bf16()
else:
    bench_float32()
    bench_ipex_bf16()