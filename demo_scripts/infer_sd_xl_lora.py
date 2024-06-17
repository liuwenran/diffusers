from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)
import torch
import os

weight_dtype = torch.float16
# pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
# pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-refiner-1.0/snapshots/5d4cfe854c9a9a87939ff3653551c2b3c99a4356'
vae_path = "madebyollin/sdxl-vae-fp16-fix"
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)

# from diffusers import DiffusionPipeline
# import torch

# pipe = DiffusionPipeline.from_pretrained(
#     "playgroundai/playground-v2.5-1024px-aesthetic",
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("cuda")

# # # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
# # from diffusers import EDMDPMSolverMultistepScheduler
# # pipe.scheduler = EDMDPMSolverMultistepScheduler()

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]

# pretrained_model_name_or_path = "playgroundai/playground-v2.5-1024px-aesthetic"

pipeline = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path, 
    vae=vae,
    torch_dtype=weight_dtype
)

# neg_embed = '/mnt/petrelfs/liuwenran/forks/diffusers/ac_neg1.safetensors'
# pipeline.load_textual_inversion(neg_embed)


# load attention processors
# lora_dir = 'work_dirs/cctv/qianqiushisong/lora-trained-xl-qianqiuhuman-e4/checkpoint-300/pytorch_lora_weights.safetensors'
# lora_dir = 'work_dirs/lora-trained-xl-fp16train3k/checkpoint-1200'
# lora_dir = 'work_dirs/lora-trained-xl-legend-deerone-e4/checkpoint-200'
# lora_dir = 'work_dirs/lora-trained-xl-legend-deerone-e4-816/checkpoint-300'
# lora_dir = 'work_dirs/lora-trained-xl-changshiyongmei-e4/checkpoint-300'
# lora_dir = 'work_dirs/lora-trained-xl-zhangqian-e4/checkpoint-300'
# lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/ClassipeintXL1.9.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/Cyberpunk _Anime_sdxl.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/chahua.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/Graphic_Portrait.safetensors'
# lora_dir = 'work_dirs/cctv/qianqiushisong3/lora-trained-xl-weiqishaonian-e4/checkpoint-300/pytorch_lora_weights.safetensors'
# lora_dir = 'work_dirs/cctv/qianqiushisong3/lora-trained-xl-dawangriji-children-e4/checkpoint-300/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/qianqiushisong3/lora-trained-xl-dawangriji-children-e4-600/checkpoint-600/pytorch_lora_weights.safetensors'
# lora_dir = 'work_dirs/cctv/lora-trained-xl-xuanwu-e4/checkpoint-300/pytorch_lora_weights.safetensors'
# lora_dir = 'work_dirs/cctv/qianqiushisong/lora-trained-xl-room-e4/checkpoint-300/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/guhua/lora-trained-xl-shisong_characters/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-shisong_characters_white_ar/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-shisong_scene/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-shisong_tools/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-shisong_scene_house4/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-shisong_character_playgroud2/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/platform/lora-trained-xl-playground_human/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/platform/lora-trained-xl-playground_human600/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-shisong_character_300/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-scene_wo_house_300/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-scene_wo_house_300_buildings/pytorch_lora_weights.safetensors'
lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/platform/lora-trained-xl-shisong_character_300_square/pytorch_lora_weights.safetensors'

pipeline.load_lora_weights(lora_dir)

pipeline = pipeline.to("cuda")

# prompt = "A photo of a chinese old man in cartoon style, Tang Dynasty, portrait, hands behind back, perfect, extremely detailed, 8k"
# prompt = 'A photo of a man riding camel in chinese ancient style, side view perspective, brown camel, perfect, extremely detailed, 8k'
# prompt = "a cartoon named changshi, A photo of an old man in chinese ancient style, Song Dynasty, without people, best quality, extremely detailed, good light"
# prompt = "a cartoon named changshi, a photo of an old man in chinese ancient style, Song Dynasty, best quality, extremely detailed, good light"
# prompt = 'oil painting, a cute girl, sunshine, best quality, perfect, extremely detailed'
# prompt = 'POP SURREALISM, a cute girl, sunshine, best quality, perfect, extremely detailed'
# prompt = 'a photo in chinese cartoon style, an old chinese officer, silver beard, white clothes, black offical hat, black boots, Tang Dynasty'
prompt = 'a photo in chinese cartoon style, an old man, Tang Dynasty, best quality, extremely detailed, perfect, 8k, masterpeice'
# prompt = 'a photo in chinese cartoon style, street in chinese ancient style,chinese ink inpainting, Song Dynasty, best quality, extremely detailed, good light'
# prompt = 'a photo in chinese cartoon style, grass land, best quality, extremely detailed, good light'
# prompt = 'a photo in chinese cartoon style, a bowl on a desk, Song Dynasty, best quality, extremely detailed, good light'
# prompt = 'a photo in chinese cartoon style, some children, school'
# prompt = 'Some dragons are circling, chinese cartoon style, Tang Dynasty, perfect, extremely detailed'
# prompt = 'a photo in chinese cartoon style, a dog, white background, whole body, best quality, extremely detailed, perfect, 8k, masterpeice'
# prompt = 'a photo in chinese cartoon style, a bowl, best quality, extremely detailed, perfect, 8k, masterpeice'
# prompt = 'From left to right, a blonde ponytail Europe girl in white shirt, a brown curly hair African girl in blue shirt printed with a bird, an Asian young man with black short hair in suit are walking in the campus happily.'
negative_prompt = 'bad light, bad hands, bad face, long body, multiple people'
# negative_prompt = 'buildings, houses'
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

folder_path = 'results/cctv_platform/character_300_square'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# img2img_ref = '/mnt/petrelfs/liuwenran/datasets/magicmaker_assert/xi/fu.png'
# from PIL import Image
# img2img_ref = Image.open(img2img_ref).convert('RGB')
# img2img_ref = img2img_ref.resize((1024, 1024))


for ind in range(10):
    # image = pipeline(prompt, negetive_prompt=negative_prompt, num_inference_steps=25, width=1080, height=1920, generator=generator).images[0]
    image = pipeline(prompt, negetive_prompt=negative_prompt, num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
    # image = pipeline(prompt, strength=1.0, image=img2img_ref).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))

