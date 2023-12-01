from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
import torch
import os
import numpy as np
import cv2
from diffusers.utils import load_image
from PIL import Image

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
pretrained_model_name_or_path = "gsdf/CounterfeitXL"

pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
)

# pipeline.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors")
pipeline = pipeline.to("cuda")

prompt = "a logo, snow mountain in background, perfect, extremely detailed, 8k"
# prompt = 'A photo of a man riding camel in chinese ancient style, side view perspective, brown camel, perfect, extremely detailed, 8k'
# prompt = 'photo of chinese young woman, highlight hair, sitting outside restaurant, wearing dress, rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores'
negative_prompt = 'disfigured, ugly, bad, immature, blur'
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

folder_path = 'results/sd_xl_counterfeit_controlnet/rangear'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


image = load_image('/mnt/petrelfs/liuwenran/datasets/rangear/logo2.jpg')
# image = image.resize((832, 1280))
# image = image.resize((1920, 960))
# image = image.resize((960, 1920))
image = image.resize((1024, 1024))

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)


for ind in range(10):
    image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=canny_image).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))

