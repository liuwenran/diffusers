import torch
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import set_seed
from diffusers import DDPMScheduler, DDIMScheduler


set_seed(2)

ckpt_path = "runwayml/stable-diffusion-v1-5"
# ckpt_path = '/nvme/liuwenran/models/huggingface/mo-di-diffusion'

pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16, revision="fp16")


pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


pipe = pipe.to("cuda")

prompt = "a building in the mountain, perfect, 8k, extremely detailed"
image = pipe(prompt).images[0]

image.save("results/output/building_ddim_eta1.png")
