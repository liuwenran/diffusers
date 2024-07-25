import torch
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import set_seed
from diffusers import DDPMScheduler, DDIMScheduler
import os

set_seed(2)

# ckpt_path = "runwayml/stable-diffusion-v1-5"
# ckpt_path = '/nvme/liuwenran/models/huggingface/mo-di-diffusion'
ckpt_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/ded79e214aa69e42c24d3f5ac14b76d568679cc2'

pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to('cuda')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

prompt = 'a beautiful chinese woman is dancing, perfect, best quality'
negative_prompt = 'bad face, bad hands'
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

folder_path = 'results/sd15/sdxl_dancing'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(10):
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))
