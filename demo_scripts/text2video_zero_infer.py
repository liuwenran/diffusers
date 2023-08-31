import torch
from diffusers import TextToVideoZeroPipeline
import imageio

# model_id = "runwayml/stable-diffusion-v1-5"
# model_id = 'resources/stable-diffusion-v1-5'
model_id = 'Linaqruf/anything-v3.0'
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None).to("cuda")

prompt = "A handsome man is playing basketball"
result = pipe(prompt=prompt).images
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("results/t2vzero/basketball4.mp4", result, fps=4)
