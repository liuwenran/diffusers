from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)
import torch
import os

weight_dtype = torch.float16
# pretrained_model_name_or_path = "/mnt/petrelfs/liuwenran/models/civitai/ckpts/copaxTimelessxlSDXL1_v8.safetensors"
# pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/models/civitai/ckpts/protovisionXLHighFidelity3D_release0630Bakedvae.safetensors'
pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/models/civitai/ckpts/leosamsHelloworldSDXL_helloworldSDXL30.safetensors'
# pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/models/civitai/ckpts/xxmix9realisticsdxl_v10.safetensors'

pipeline = StableDiffusionXLPipeline.from_single_file(pretrained_model_name_or_path, torch_dtype=weight_dtype)
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
# lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/ClassipeintXL1.9.safetensors'
lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/nudify_xl_lite.safetensors'
pipeline.load_lora_weights(lora_dir)

# lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/pantyhose_xl_v1.safetensors'
# pipeline.load_lora_weights(lora_dir)


pipeline = pipeline.to("cuda")

# prompt = "a chinese girl, smiling, cute, sunshine, perfect, extremely detailed, 8k"
# prompt = 'a chinese village, sunset, big view'
# prompt = 'A photo of a man riding camel in chinese ancient style, side view perspective, brown camel, perfect, extremely detailed, 8k'
# prompt = 'photo of chinese young woman, highlight hair, sitting outside restaurant, wearing dress, rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores'
# prompt = 'oil painting, a chinese old house, sunshine, perfect, extremely detailed'
# prompt = 'a beautiful chinese woman, smiling, thin, pantyhose, black pantyhose, white sweater, black short skirt, black gloves, full-body photo, standing on snow ground, Gorgeous, charming, attractive, best quality, extremely detailed, sharp foucus, masterpeice, 8k, photorealistic, awesome, perfect'
# prompt = 'leogirl,(pale skin, blush), (snow on stomach, snow on breasts, snow on legs), cold, highly detailed realistic photo of chinese beautiful (skinny) smiling woman standing in front of the (x-mas tree in busy city street:1.05), new year theme, Santa hat, nude and barefoot, (full body, feet), (winter, snow, frost, snowstorm:1.2), foggy evening, best quality, 8k, contrast lighting, detailed background,shot by iphone,'
# prompt = 'leogirl,(pale skin, blush),pink swimsuit, highly detailed realistic photo of chinese beautiful (skinny) smiling woman standing outside, barefoot, (full body, feet), (winter, snow, snow mountain), best quality, 8k, contrast lighting, detailed background,shot by iphone,'
prompt = 'leogirl, (pale skin, blush), black pantyhose, black bra, black skirt, highly detailed realistic photo of chinese beautiful (skinny) smiling woman standing outside, santa hat, (full body, feet), (winter, snow, snow mountain), best quality, 8k, contrast lighting, detailed background,shot by iphone,'


# negative_prompt = 'disfigured, ugly, bad, immature, cartoon, 3d, painting, b&w'
# negative_prompt = 'disfigured, ugly,bad legs, bad body, bad face, bad anatomy, bad hands, blur, painting, lowres,low quality, watermark, render, CG'
negative_prompt = '(worst quality,low resolution,bad hands,open mouth),distorted,twisted,watermark, disfigured, ugly,bad legs, bad body, bad face, bad anatomy, bad hands'

generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

folder_path = 'results/sd_xl_3d_copax/skiing_white_black_nudify_goodp_santa'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(20):
    image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))

