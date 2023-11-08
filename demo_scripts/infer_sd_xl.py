from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)
import torch
import os

weight_dtype = torch.float32
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
vae_path = "madebyollin/sdxl-vae-fp16-fix"
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path, vae=vae, torch_dtype=weight_dtype
)

scheduler_args = {}

if "variance_type" in pipeline.scheduler.config:
    variance_type = pipeline.scheduler.config.variance_type

    if variance_type in ["learned", "learned_range"]:
        variance_type = "fixed_small"

    scheduler_args["variance_type"] = variance_type

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)


pipeline = pipeline.to("cuda")

prompt = "A photo of shanghai in cartoon style, Tang Dynasty, portrait, perfect, extremely detailed, 8k"
# prompt = 'A photo of a man riding camel in chinese ancient style, side view perspective, brown camel, perfect, extremely detailed, 8k'
prompt = 'photo of chinese young woman, highlight hair, sitting outside restaurant, wearing dress, rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores'
negative_prompt = 'disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w'
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

folder_path = 'results/sd_xl/magicmaker_assert_young'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(20):
    image = pipeline(prompt, num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))

