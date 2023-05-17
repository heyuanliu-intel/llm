import time
from diffusers import StableDiffusionPipeline

prompt = "sailing ship in storm by Rembrandt"
# model_id = "runwayml/stable-diffusion-v1-5"
model_id = "/home/heyuan/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a/"
nb_pass = 10

# build a StableDiffusionPipeline with the default float32 data type
pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")


def elapsed_time(pipeline, nb_pass=10, num_inference_steps=20):
    start = time.time()
    for _ in range(nb_pass):
        _ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np")
    end = time.time()
    return (end - start) / nb_pass


# warmup
images = pipe(prompt, num_inference_steps=10).images

time_original_model = elapsed_time(pipe, nb_pass=nb_pass)

print(time_original_model)
