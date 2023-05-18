import time
from optimum.intel.openvino import OVStableDiffusionPipeline

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


def bench_ov_bf16():
    print("benchmark standard pipeline with openvino and bf16")
    pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
    # warmup
    images = pipe(prompt, num_inference_steps=10).images
    # time_ov_model_bf16 = elapsed_time(pipe)
    pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
    images = pipe(prompt, num_inference_steps=10).images
    latency = elapsed_time(pipe)
    print(f"latency is {latency}")


bench_ov_bf16()
