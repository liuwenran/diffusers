# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image

MIN_IND = 54
MAX_IND = 55
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

lora_dir = 'work_dirs/lora-trained-xl-qianqiuhuman-e4/checkpoint-300'
pipeline.load_lora_weights(lora_dir)
pipeline = pipeline.to("cuda")

# prompt
lora_trigger = 'a photo in cartoon style, '


generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

# role
# role = '/mnt/petrelfs/liuwenran/datasets/角色视图/dongtinglan.txt'
role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/角色视图/images.txt'

prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/角色视图/prompts.txt'
prompt_lines = open(prompt_file, 'r').readlines()
prompt_dict = {}
for line in prompt_lines:
    line = line.strip()
    line_role = line.split(':')[0]
    line_prompt = line.split(':')[1]
    prompt_dict[line_role] = line_prompt


lines = open(role, 'r').readlines()

for ind, line in enumerate(lines):
    if ind >= MIN_IND and ind < MAX_IND:
        print('image ind ' + str(ind))
        line = line.strip()
        print(line)
        img_name = line.split('/')[-1].split('.')[0]

        # prompt =  'an old chinese ancient officer, silver beard, Tang Dynasty, white clothes, black offical hat, black boots'
        prompt = None
        for role in prompt_dict.keys():
            if role in img_name:
                prompt = prompt_dict[role]
                break

        prompt = 'fat woman back,' + prompt
        if '背' in line:
            prompt = prompt + ',back view, '
        elif '左' in line or '右' in line:
            prompt = prompt + ',side view,'
        elif '正' in line:
            prompt = prompt + ',front view,'
        prompt = lora_trigger + prompt
        print(prompt)

        image = load_image(line)
        image = image.resize((960, 1920))

        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        folder_path = 'results/qianxiugesong/lora-trained-xl-qianqiuhuman-e4-0.6-fp16'
        folder_path = os.path.join(folder_path, img_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for ind in range(10):
            image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=canny_image).images[0]
            image.save(os.path.join(folder_path, str(ind) + ".png"))


