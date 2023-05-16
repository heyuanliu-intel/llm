import torch
import intel_extension_for_pytorch as ipex
from diffusers import StableDiffusionPipeline
import time
prompt = "sailing ship in storm by Rembrandt"
model_id = "runwayml/stable-diffusion-v1-5"
# Help function for time evaluation


def elapsed_time(pipeline, nb_pass=3, num_inference_steps=20):
    # warmup
    for _ in range(2):
        images = pipeline(
            prompt, num_inference_steps=num_inference_steps).images
    # time evaluation
    start = time.time()
    for _ in range(nb_pass):
        pipeline(prompt, num_inference_steps=num_inference_steps)
    end = time.time()
    return (end - start) / nb_pass


##############     bf16 inference performance    ###############
# 1.IPEX Pipeline initialization
pipe = DiffusionPipeline.from_pretrained(
    model_id, custom_pipeline="stable_diffusion_ipex")
pipe.prepare_for_ipex(prompt, infer_type='bf16')
# 2.Original Pipeline initialization
pipe2 = StableDiffusionPipeline.from_pretrained(model_id)
# 3.Compare performance between Original Pipeline and IPEX Pipeline
with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    latency = elapsed_time(pipe)
    print("Latency of StableDiffusionIPEXPipeline--bf16", latency)
    latency = elapsed_time(pipe2)
    print("Latency of StableDiffusionPipeline--bf16", latency)
##############     fp32 inference performance    ###############
# 1.IPEX Pipeline initialization
pipe3 = DiffusionPipeline.from_pretrained(
    model_id, custom_pipeline="stable_diffusion_ipex")
pipe3.prepare_for_ipex(prompt, infer_type='fp32')
# 2.Original Pipeline initialization
pipe4 = StableDiffusionPipeline.from_pretrained(model_id)
# 3.Compare performance between Original Pipeline and IPEX Pipeline
with torch.no_grad():
    latency = elapsed_time(pipe3)
    print("Latency of StableDiffusionIPEXPipeline--fp32", latency)
    latency = elapsed_time(pipe4)
    print("Latency of StableDiffusionPipeline--fp32", latency)
