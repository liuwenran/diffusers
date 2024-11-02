# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, StableDiffusionXLPipeline
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image
import argparse

# image = load_image('/mnt/petrelfs/liuwenran/datasets/shanhaibailing/control/giraffe.jpg')
# image = image.crop((0, 0, 1024, 1024))
parser = argparse.ArgumentParser()
parser.add_argument("--min", type=int)
parser.add_argument("--max", type=int)
parser.add_argument("--type", type=int)
args = parser.parse_args()

# initialize the models and pipeline
weight_dtype = torch.float16

controlnet_conditioning_scale = 0.4  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=weight_dtype
)

vae_path = "madebyollin/sdxl-vae-fp16-fix"
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)

if args.type == 0:

# pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
    )
    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     pretrained_model_name_or_path, vae=vae, torch_dtype=weight_dtype
    # )

else:
    single_file_path = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/ckpts/AnythingXL_xl.safetensors'
    pipeline = StableDiffusionXLControlNetPipeline.from_single_file(single_file_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype)


lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/pet/style_jianbi/pytorch_lora_weights.safetensors'


if args.type == 0:
    pipeline.load_lora_weights(lora_dir)

pipeline = pipeline.to("cuda")

negative_prompt = 'blur, low quality'
lora_trigger = 'a photo in stick drawings style, cute pet'



generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)


image_lines = open('/mnt/petrelfs/liuwenran/datasets/petdata/content/cat.txt').read().splitlines()


root_path = 'results/pet/'

init_prompt = lora_trigger
if not os.path.exists(root_path):
    os.makedirs(root_path)

for ind, line in enumerate(image_lines):
    if ind >= args.min and ind < args.max:
        image = load_image(line)
        image = image.convert("RGB")
        image_size = image.size
        if image_size[0] < 512 or image_size[1] < 512:
            new_size = (image_size[0] * 2, image_size[1] * 2)
            image = image.resize(new_size)
        elif image_size[0] > 2048 or image_size[1] > 2048:
            new_size = (int(image_size[0] / 2), int(image_size[1] / 2))
            image = image.resize(new_size)
        # image = image.resize((832, 1280))
        # image = image.resize((1920, 1080))
        # image = image.resize((960, 1920))
        # image = image.resize((1024, 1024))
        # image = image.resize((800, 1024))
        # image = image.resize((800, 1200))
        # image = image.crop((0, 200, 1080, 1600))

        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        folder_path = os.path.join(root_path, line.split('/')[-1].split('.')[0])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # prompt = init_prompt + prompt_lines[ind]
        prompt = init_prompt  + ', cat'

        print(f'ind {ind} prompt: {prompt}')
        for ind in range(10):        
            image = pipeline(prompt, negative_prompt=negative_prompt,controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=canny_image).images[0]
            image.save(os.path.join(folder_path, str(ind) + ".png"))