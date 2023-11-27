from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)
import torch
import os

weight_dtype = torch.float32
# pretrained_model_name_or_path = "/mnt/petrelfs/liuwenran/models/civitai/ckpts/copaxTimelessxlSDXL1_v8.safetensors"
pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/models/civitai/ckpts/protovisionXLHighFidelity3D_release0630Bakedvae.safetensors'

pipeline = StableDiffusionXLPipeline.from_single_file(pretrained_model_name_or_path)
# vae_path = "madebyollin/sdxl-vae-fp16-fix"
# vae = AutoencoderKL.from_pretrained(
#     vae_path,
#     torch_dtype=weight_dtype,
# )
# pipeline = StableDiffusionXLPipeline.from_pretrained(
#     pretrained_model_name_or_path, vae=vae, torch_dtype=weight_dtype
# )

# scheduler_args = {}

# if "variance_type" in pipeline.scheduler.config:
#     variance_type = pipeline.scheduler.config.variance_type

#     if variance_type in ["learned", "learned_range"]:
#         variance_type = "fixed_small"

#     scheduler_args["variance_type"] = variance_type

# pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/ClassipeintXL1.9.safetensors'
pipeline.load_lora_weights(lora_dir)


pipeline = pipeline.to("cuda")

# prompt = "a chinese girl, smiling, cute, sunshine, perfect, extremely detailed, 8k"
# prompt = 'a chinese village, sunset, big view'
# prompt = 'A photo of a man riding camel in chinese ancient style, side view perspective, brown camel, perfect, extremely detailed, 8k'
# prompt = 'photo of chinese young woman, highlight hair, sitting outside restaurant, wearing dress, rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores'
prompt = 'oil painting, a chinese old house, sunshine, perfect, extremely detailed'
# negative_prompt = 'disfigured, ugly, bad, immature, cartoon, 3d, painting, b&w'
negative_prompt = 'disfigured, ugly, bad, blur'
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

folder_path = 'results/sd_xl_3d_oil/old_house'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(10):
    image = pipeline(prompt, num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))

