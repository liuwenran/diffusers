import torch
import os
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
# url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"


# prompt = ''
prompt = "chinese ink painting, white background"
# prompt += 'high resolution, perfect, extremely detailed, 8k, masterpeice, good light'
negative_prompt = 'blur, bad light'

image_folder_path = '/mnt/petrelfs/liuwenran/forks/diffusers/results/sd_xl_dreambooth_lora_canny/lora-trained-xl-gufengstreet-e4_white'
folder_path = 'results/refine/lora-trained-xl-gufengstreet-e4_white'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

for i in range(10):
    image_path = os.path.join(image_folder_path, str(i) + '.png')
    init_image = load_image(image_path).convert("RGB")
    image = pipe(prompt, negative_prompt=negative_prompt, image=init_image, num_inference_steps=20).images[0]
    image.save(os.path.join(folder_path, str(i) + '.png'))

# image_path = '/mnt/petrelfs/liuwenran/forks/diffusers/results/sd_xl_dreambooth_lora/deerone-e4_dragon_song/5.png'
# init_image = load_image(image_path).convert("RGB")
# image = pipe(prompt, negative_prompt=negative_prompt, image=init_image).images[0]
# image.save('results/refine/deerone-e4_dragon_song_5.png')
