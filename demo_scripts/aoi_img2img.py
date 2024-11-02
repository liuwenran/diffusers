from PIL import Image
import os
from diffusers import StableDiffusionXLImg2ImgPipeline
import numpy as np
from diffusers.training_utils import set_seed

set_seed(20231)

# load the pipeline
ckpt_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--gsdf--CounterfeitXL/snapshots/4708675873bd09833aabc3fd4cb2de5fcd1726ac'

device = "cuda"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(ckpt_path).to(device)

img = '/mnt/petrelfs/liuwenran/datasets/petdata/content/cat/loki.jpg'
character_name = img.split('/')[-1].split('.')[0]
init_image = Image.open(img)
init_image = init_image.resize((1024, 1024)).convert('RGB')

prompt = "a cute cat in cartoon style, best quality, perfect, extremely detailed, 8k"
negative_prompt = "ugly, bad, disfigured, blur"

for strength in np.arange(0.6, 0.7, 0.01):
    folder_path = f"results/aoi/{character_name}-{strength:.2f}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i in range(2):
        image = pipe(prompt, negative_prompt=negative_prompt, image=init_image, strength=strength).images[0]
        image.save(os.path.join(folder_path, f"{i}.png"))