import torch
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import set_seed


set_seed(1)

ckpt_path = 'resources/stable-diffusion-v1-5'
# ckpt_path = '/nvme/liuwenran/models/huggingface/mo-di-diffusion'

pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

prompt = "modern disney style "
image = pipe(prompt).images[0]

image.save("resources/output/sd-modi.png")
