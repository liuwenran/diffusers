from diffusers import (
    AutoencoderKL,
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


# load attention processors
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
lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/qianqiushisong3/lora-trained-xl-dawangriji-children-e4-600/checkpoint-600/pytorch_lora_weights.safetensors'
pipeline.load_lora_weights(lora_dir)

pipeline = pipeline.to("cuda")

# prompt = "A photo of a chinese old man in cartoon style, Tang Dynasty, portrait, hands behind back, perfect, extremely detailed, 8k"
# prompt = 'A photo of a man riding camel in chinese ancient style, side view perspective, brown camel, perfect, extremely detailed, 8k'
# prompt = "a cartoon named changshi, A photo of an old man in chinese ancient style, Song Dynasty, without people, best quality, extremely detailed, good light"
# prompt = "a cartoon named changshi, a photo of an old man in chinese ancient style, Song Dynasty, best quality, extremely detailed, good light"
# prompt = 'oil painting, a cute girl, sunshine, best quality, perfect, extremely detailed'
# prompt = 'POP SURREALISM, a cute girl, sunshine, best quality, perfect, extremely detailed'
prompt = 'a photo in chinese cartoon style, an old chinese officer, silver beard, white clothes, black offical hat, black boots'
# prompt = 'a photo in chinese cartoon style, an old man'
# prompt = 'a photo in chinese cartoon style, some children, school'
negative_prompt = 'bad light, bad hands, bad face, yellow background'
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

folder_path = 'results/sd_xl_lora/dawangriji-children-600'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(10):
    image = pipeline(prompt, num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))

