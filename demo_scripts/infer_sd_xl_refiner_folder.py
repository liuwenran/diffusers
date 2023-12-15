import torch
import os
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import glob


def find_images(folder_path):
    image_extensions = ['jpg', 'jpeg', 'png', 'gif']  # 可能的图片文件扩展名

    images = []
    for ext in image_extensions:
        search_pattern = os.path.join(folder_path, f'*.{ext}')
        images.extend(glob.glob(search_pattern))

    return images


pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
# url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"


prompt = 'a beautiful chinese woman, smiling, thin, pantyhose, black pantyhose, white sweater, black short skirt, black gloves, snow mountain in the background, Gorgeous, charming, attractive, best quality, extremely detailed, sharp foucus, masterpeice, 8k, photorealistic, awesome, perfect'
negative_prompt = 'disfigured, ugly,bad legs, bad body, bad face, bad anatomy, bad hands, blur, painting, lowres,low quality, watermark, render, CG'

source_folder_path = '/mnt/petrelfs/liuwenran/forks/diffusers/results/sd_xl_3d_copax/skiing_white_black'
folder_path = '/mnt/petrelfs/liuwenran/forks/diffusers/results/sd_xl_3d_copax/skiing_white_black_refine'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

images = find_images(source_folder_path)

for img in images:
    img_name = img.split('/')[-1]
    init_image = load_image(img).convert("RGB")
    image = pipe(prompt, negative_prompt=negative_prompt, image=init_image, strength=0.7).images[0]
    image.save(os.path.join(folder_path, img_name))

# image_path = '/mnt/petrelfs/liuwenran/forks/diffusers/results/sd_xl_dreambooth_lora/deerone-e4_dragon_song/5.png'
# init_image = load_image(image_path).convert("RGB")
# image = pipe(prompt, negative_prompt=negative_prompt, image=init_image).images[0]
# image.save('results/refine/deerone-e4_dragon_song_5.png')
