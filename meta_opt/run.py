import torch
from transformers import pipeline

# This works on a base Colab instance.
# Pick a larger checkpoint if you have time to wait and enough disk space!
checkpoint = "facebook/opt-6.7b"
generator = pipeline("text-generation", model=checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

# Perform inference
token = generator("I like")
print(f"{token}")