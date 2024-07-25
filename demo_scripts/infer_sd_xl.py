from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)
import torch
import os

weight_dtype = torch.float16
# pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
vae_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15'
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)
pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--playgroundai--playground-v2.5-1024px-aesthetic/snapshots/1e032f13f2fe6db2dc49947dbdbd196e753de573'
pipeline = StableDiffusionXLPipeline.from_pretrained(
    # pretrained_model_name_or_path, vae=vae, torch_dtype=weight_dtype
    pretrained_model_name_or_path, torch_dtype=weight_dtype
)


pipeline = pipeline.to("cuda")

# prompt = "A photo of shanghai in cartoon style, Tang Dynasty, portrait, perfect, extremely detailed, 8k"
# prompt = 'A photo of a man riding camel in chinese ancient style, side view perspective, brown camel, perfect, extremely detailed, 8k'
# prompt = 'photo of chinese young woman, highlight hair, sitting outside restaurant, wearing dress, rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores'
# negative_prompt = 'disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w'
prompt = 'a beautiful chinese woman is dancing, full body, perfect, best quality'
negative_prompt = 'bad face, bad hands'

folder_path = 'results/sd_xl/playgroundv25_dancing'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(10):
    generator = torch.Generator(device=torch.device('cuda')).manual_seed(ind)
    image = pipeline(prompt, num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))

