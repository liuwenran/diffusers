# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image, ImageOps

# image = load_image('/mnt/petrelfs/liuwenran/datasets/shanhaibailing/control/giraffe.jpg')
# image = image.crop((0, 0, 1024, 1024))
from controlnet_aux import OpenposeDetector

weight_dtype = torch.float16

# Compute openpose conditioning image.
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)
openpose_image = openpose(image)

import ipdb;ipdb.set_trace();
tpose_image = load_image('/mnt/petrelfs/liuwenran/datasets/magicmaker_assert/t-pose/padded.jpg')


# 设置填充的大小和颜色
# padding_size = (1280 - 1152) // 2
# padding_color = (0, 0, 0)

# # 进行填充
# padded_image = ImageOps.expand(tpose_image, padding_size, padding_color)
# padded_image.save('/mnt/petrelfs/liuwenran/datasets/magicmaker_assert/t-pose/padded.jpg')

# import ipdb;ipdb.set_trace();

# Initialize ControlNet pipeline.
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)

# # initialize the models and pipeline

controlnet_conditioning_scale = 0.8  # recommended for good generalization
# controlnet = ControlNetModel.from_pretrained(
#     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=weight_dtype
# )

vae_path = "madebyollin/sdxl-vae-fp16-fix"
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
# pretrained_model_name_or_path = 'gsdf/CounterfeitXL'
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
)
# pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
#     pretrained_model_name_or_path, controlnet=controlnet, torch_dtype=weight_dtype
# )


# pipeline.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors")

# cctv/lora-trained-xl-comic-e4/checkpoint-300'
# lora_dir = 'work_dirs/cctv/lora-trained-xl-3dstyle-e4/checkpoint-300'

lora_dir = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/minimalism.safetensors'
pipeline.load_lora_weights(lora_dir)

pipeline = pipeline.to("cuda")



# t2i_pipe = StableDiffusionXLPipeline(pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2,
#                                      pipeline.tokenizer, pipeline.tokenizer_2, pipeline.unet, pipeline.scheduler)
# pipeline = t2i_pipe

# init_image = load_image('/mnt/petrelfs/liuwenran/forks/diffusers/results/multipipeline/girl_for_animation/6.png').convert("RGB")
# prompt = "a girl is standing"
# img2img_pipeline = StableDiffusionXLImg2ImgPipeline(pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2,
#                                                     pipeline.tokenizer, pipeline.tokenizer_2, pipeline.unet, pipeline.scheduler)
# pipeline = img2img_pipeline


# prompt = 'in Tang Dynasty, handsome, brown and beige clothes, white boots.'
# prompt = 'A young Chinese ancient male, in Tang Dynasty, tall and thin body type, very handsome, white official hat, white clothes with pattern, black boots.'
# # lora_trigger = 'a photo of a boy in cartoon style, standing in front of a house'
# lora_trigger = 'a photo of a man in cartoon style, standing in front of a house, '
# prompt = lora_trigger + prompt
prompt = 'a girl, beautiful, short skirt, T-shirt, smiling, best quality, extremely detailed, 8k'

generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

# get canny image
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/donghuatest/content/xiaohai.png')
image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/donghuatest/content/libai.png')
image = image.resize((960, 1920))

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

folder_path = 'results/multipipeline/girl_for_animation_tpose_test'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(10):
    # image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=canny_image).images[0]
    image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=tpose_image.resize((1024, 1024))).images[0]
    # image = pipeline(prompt, num_inference_steps=25, width=1024, height=1024).images[0]
    # image = pipeline(prompt, image=init_image).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))
