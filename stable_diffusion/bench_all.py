import time
import intel_extension_for_pytorch as ipex
import torch
import argparse
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from stable_diffusion_ipex import StableDiffusionIPEXPipeline
from optimum.intel.openvino import OVStableDiffusionPipeline

# model_id = "runwayml/stable-diffusion-v1-5"
# Use cached folder to avoid download the models from network
model_id = "/home/heyuan/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a/"
prompt = "sailing ship in storm by Rembrandt"
nb_pass = 10

with_compile = True


def compile_pipe(pipe):
    if with_compile:
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        return pipe
    else:
        return pipe


def elapsed_time(pipeline, nb_pass=10, num_inference_steps=20):
    start = time.time()
    for _ in range(nb_pass):
        _ = pipeline(prompt, num_inference_steps=num_inference_steps, height=512, width=512)
    end = time.time()
    return (end - start) / nb_pass


def bench_float32():
    print("benchmark standard pipeline with float32")
    # build a StableDiffusionPipeline with the default float32 data type
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")
    pipe = compile_pipe(pipe)

    with torch.no_grad():
        # warmup
        images = pipe(prompt, num_inference_steps=10).images
        # benchmarking
        latency = elapsed_time(pipe, nb_pass=nb_pass)
        print(f"latency is {latency}")


def bench_bf16():
    print("benchmark standard pipeline with bf16")
    # build a StableDiffusionPipeline with the default float32 data type
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")
    pipe = compile_pipe(pipe)

    with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        # warmup
        images = pipe(prompt, num_inference_steps=10).images
        # benchmarking
        latency = elapsed_time(pipe, nb_pass=nb_pass)
        print(f"latency is {latency}")


def bench_ipex_bf16():
    print("benchmark standard pipeline with ipex and bf16")
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

    pipe = compile_pipe(pipe)

    with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        latency = elapsed_time(pipe, nb_pass=nb_pass)
        print(f"latency is {latency}")


def bench_ipex_scheduler_bf16():
    print("benchmark standard pipeline with ipex, bf16 and custom scheduler")
    dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=dpm)

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

    pipe = compile_pipe(pipe)

    with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        latency = elapsed_time(pipe, nb_pass=nb_pass)
        print(f"latency is {latency}")


def bench_ipex_custom_fp32():
    print("benchmark standard pipeline with ipex fp32 and customized pipeline")
    pipe = StableDiffusionIPEXPipeline.from_pretrained(model_id)
    pipe.prepare_for_ipex(prompt, dtype=torch.float32, height=512, width=512)
    pipe = compile_pipe(pipe)

    with torch.no_grad():
        latency = elapsed_time(pipe)
        print(f"latency is {latency}")


def bench_ipex_custom_bf16():
    print("benchmark standard pipeline with ipex bf16 and customized pipeline")
    pipe = StableDiffusionIPEXPipeline.from_pretrained(model_id)
    pipe.prepare_for_ipex(prompt, dtype=torch.bfloat16, height=512, width=512)
    pipe = compile_pipe(pipe)

    with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        latency = elapsed_time(pipe)
        print(f"latency is {latency}")


def bench_ov_bf16():
    print("benchmark standard pipeline with openvino and bf16")
    pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
    # warmup
    images = pipe(prompt, num_inference_steps=10).images
    # time_ov_model_bf16 = elapsed_time(pipe)
    pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
    with torch.no_grad():
        images = pipe(prompt, num_inference_steps=10).images
        latency = elapsed_time(pipe)
        print(f"latency is {latency}")


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--base_fp32', action='store_true', help='baseline with FP32 benchmarking')
parser.add_argument('--base_bf16', action='store_true', help='baseline with BF16 benchmarking')
parser.add_argument('--ipex_bf16', action='store_true', help='benchmarking with ipex and bf16')
parser.add_argument('--ipex_scheduler_bf16', action='store_true', help='benchmarking with ipex and bf16')
parser.add_argument('--ipex_custom_fp32', action='store_true', help='benchmarking with ipex and fp32')
parser.add_argument('--ipex_custom_bf16', action='store_true', help='benchmarking with ipex and bf16')
parser.add_argument('--ov_bf16', action='store_true', help='benchmarking with openvino and bf16')
args = parser.parse_args()

if args.base_fp32:
    bench_float32()
elif args.base_bf16:
    bench_bf16()
elif args.ipex_bf16:
    bench_ipex_bf16()
elif args.ipex_scheduler_bf16:
    bench_ipex_scheduler_bf16()
elif args.ipex_custom_fp32:
    bench_ipex_custom_fp32()
elif args.ipex_custom_bf16:
    bench_ipex_custom_bf16()
elif args.ov_bf16:
    bench_ov_bf16()
else:
    bench_float32()
    bench_bf16()
    bench_ipex_bf16()
    bench_ipex_scheduler_bf16()
    bench_ov_bf16()
    bench_ipex_custom_fp32()
    bench_ipex_custom_bf16()
