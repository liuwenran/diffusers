# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image

MIN_IND = 0
MAX_IND = 20
SNOW_IND_LIST = [3, 8, 9, 13]
NIGHT_IND_LIST = [2, 7, 10, 11, 12]
print(f'MIN {MIN_IND} MAX {MAX_IND}')

# initialize the models and pipeline
weight_dtype = torch.float16

controlnet_conditioning_scale = 1.0  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=weight_dtype
)

vae_path = "madebyollin/sdxl-vae-fp16-fix"
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
)

pipeline.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors")

lora_dir = 'work_dirs/cctv/lora-trained-xl-hutaotao-e4/checkpoint-300'
pipeline.load_lora_weights(lora_dir)
pipeline = pipeline.to("cuda")

# prompt
lora_trigger = 'a photo of day scene in cartoon style, ink painting,'


generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

scene = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/scenes/day_imgs.txt'

scene_prompt = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/scenes/prompts_day.txt'

prompt_lines = open(scene_prompt, 'r').readlines()

lines = open(scene, 'r').readlines()

for ind, line in enumerate(lines):
    # if ind in NIGHT_IND_LIST:
    if ind >= 6:
        print('image ind ' + str(ind))
        line = line.strip()
        print(line)
        img_name = line.split('/')[-1].split('.')[0]

        # prompt =  'an old chinese ancient officer, silver beard, Tang Dynasty, white clothes, black offical hat, black boots'
        # prompt = prompt_lines[ind].strip().split(':')[1]
        prompt = prompt_lines[ind].strip()
        # prompt = ',ink painting, floating structures, multilayered realism, stylized realism, ultra wide view, high angle view, A street corner in Chang an City, in Tang Dynasty, winter, cinematic lighting, Chinese Landscape painting, exquisite detail, freehand brushwork, With an adorable simplistic style and elements of Chinese ink painting and calligraphy, the overall aesthetic is reminiscent of traditional Chinese brush and ink paintings'
        # prompt = ''
        prompt = lora_trigger + prompt
        print(prompt)

        image = load_image(line)
        width, height = image.size
        new_width, new_height = None, None
        if width > 2048:
            new_width = 2048
            new_height = int(2048 / width * height)
            image = image.resize((new_width, new_height))

        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        folder_path = 'results/t2i_sd_xl_lora_canny/donghuatest/lora-trained-xl-hutaotao-e4-checkpoint-300'
        folder_path = os.path.join(folder_path, img_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for ind in range(10):
            image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, negative_prompt='river', num_inference_steps=25, image=canny_image).images[0]
            image.save(os.path.join(folder_path, str(ind) + ".png"))


