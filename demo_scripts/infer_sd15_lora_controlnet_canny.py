# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image

# image = load_image('/mnt/petrelfs/liuwenran/datasets/shanhaibailing/control/giraffe.jpg')
# image = image.crop((0, 0, 1024, 1024))


# initialize the models and pipeline
weight_dtype = torch.float32

controlnet_conditioning_scale = 0.5  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype
)

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, controlnet=controlnet, torch_dtype=weight_dtype
)

# load attention processors
# lora_dir = 'lora-trained-xl-fp16train3k'
# lora_dir = 'lora-trained-xl-fp16train2k'
# lora_dir = 'lora-trained-xl-fp16train3k/checkpoint-1200'
# lora_dir = 'work_dirs/lora-trained-xl-legend-onedeer/checkpoint-1400'
# lora_dir = 'work_dirs/lora-trained-xl-legend-deerone-e4-816/checkpoint-200'
# lora_dir = 'work_dirs/lora-trained-xl-zhangqian-e4/checkpoint-300'
# lora_dir = 'loras/chinese_peking_opera.safetensors'
# lora_dir = 'loras/qmpeony-sdxl_v1.safetensors'
# lora_dir = 'loras/Southern-Spring-sdxl_v1.safetensors'
# lora_dir = 'loras/landscape-painting-sdxl_v2.safetensors'
lora_dir = 'sd_15_loras/ancient_chinese_indoors.safetensors'
# lora_dir = 'sd_15_loras/daji_v2.safetensors'
# lora_dir = 'sd_15_loras/MoXinV1.safetensors'
# lora_dir = 'sd_15_loras/shanshui-000004.safetensors'
# lora_dir = 'sd_15_loras/wyy-000009.safetensors'
pipeline.load_lora_weights(lora_dir)

pipeline = pipeline.to("cuda")

# prompt = "A photo of a chinese old man in cartoon style, Tang Dynasty, portrait, without hands, perfect, extremely detailed, 8k"
# prompt = "A photo of chinese buildings in cartoon style, chinese ancient, Song Dynasty, without people, best quality, extremely detailed, good light"
prompt = "A photo of chinese buildings in cartoon style, chinese ancient, Song Dynasty, without people, best quality, extremely detailed, good light"

# lora_trigger = 'liujiyou, Chinese ink painting, '
# lora_trigger = 'chinese peking opera '
# lora_trigger = 'QIEMANCN, '
# lora_trigger = 'QIEM6NCN, '
lora_trigger = 'ancient_chinese_indoors'
# lora_trigger = 'daji'
# lora_trigger = 'shuimobysim'
# lora_trigger = ''
# lora_trigger = 'wyy_style'

prompt = lora_trigger + prompt

generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

# get canny image
image = load_image('/mnt/petrelfs/liuwenran/datasets/sichouzhilu/huizhan_1920.jpg')
image = image.resize((960, 540))

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

folder_path = 'results/sd_15_lora_canny/xian_huizhan_sd15_indoors_lora0.5'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(10):
    image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=canny_image).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))


