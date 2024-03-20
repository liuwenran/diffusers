# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image

# image = load_image('/mnt/petrelfs/liuwenran/datasets/shanhaibailing/control/giraffe.jpg')
# image = image.crop((0, 0, 1024, 1024))


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
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
)

# pipeline.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors")

# trained_ckpt = 'work_dirs/t2i-changshiban/t2i-changshiban-trainckpt-e5-fp16'
# pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
#     trained_ckpt, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
# )
# pipeline.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors")

# load attention processors
# lora_dir = 'work_dirs/cctv/qianqiushisong/lora-trained-xl-qianqiuhuman-e4/checkpoint-300/pytorch_lora_weights.safetensors'

# lora_dir = 'lora-trained-xl-fp16train3k'
# lora_dir = 'lora-trained-xl-fp16train2k'
# lora_dir = 'lora-trained-xl-fp16train3k/checkpoint-1200'
# lora_dir = 'work_dirs/lora-trained-xl-legend-onedeer/checkpoint-1400'
# lora_dir = 'work_dirs/lora-trained-xl-legend-deerone-e4-816/checkpoint-200'
# lora_dir = 'work_dirs/lora-trained-xl-zhangqian-e4/checkpoint-300'
# lora_dir = 'work_dirs/lora-trained-xl-dunhuangbihua-e4/checkpoint-300'
# lora_dir = 'loras/chinese_peking_opera.safetensors'
# lora_dir = 'loras/qmpeony-sdxl_v1.safetensors'
# lora_dir = 'loras/Southern-Spring-sdxl_v1.safetensors'
# lora_dir = 'loras/landscape-painting-sdxl_v2.safetensors'
# lora_dir = 'work_dirs/lora-trained-xl-gufengperson-e4/checkpoint-300'
# lora_dir = 'work_dirs/lora-trained-xl-gufengstreet-e4-640/checkpoint-300'
lora_dir = 'work_dirs/lora-trained-xl-xuanwu-e4/checkpoint-300'
# lora_dir = 'work_dirs/lora-trained-xl-ancientpic-e4/checkpoint-300'
# lora_dir = 'work_dirs/t2i-changshiban-e4-fp16/checkpoint-2700'
# lora_dir = 'work_dirs/t2i-changshiban/t2i-changshiban-fullsize720-e4/checkpoint-3400'
# lora_dir = 'work_dirs/cctv/lora-trained-xl-3dstyle-e4/checkpoint-300'
# lora_dir = 'work_dirs/cctv/lora-trained-xl-comic-e4/checkpoint-300'
# lora_dir = 'work_dirs/cctv/lora-trained-xl-leyuan-e4/checkpoint-300'

pipeline.load_lora_weights(lora_dir)

pipeline = pipeline.to("cuda")

# prompt = "A photo of a chinese old man in cartoon style, Tang Dynasty, portrait, without hands, perfect, extremely detailed, 8k"
# prompt = "A photo of chinese buildings in cartoon style, chinese ancient, Song Dynasty, without people, best quality, extremely detailed, good light"
# prompt = "A photo of chinese buildings in cartoon style, chinese ancient, Song Dynasty, without people, best quality, extremely detailed, good light"
# prompt = "A photo of a man in chinese ancient style, chinese ink inpainting, Song Dynasty, without people, best quality, extremely detailed, good light"
# prompt = "A photo of a man in chinese ancient style, chinese ink inpainting, Song Dynasty, best quality, extremely detailed, good light"
# prompt = "A photo of a man in chinese ancient style, Tang Dynasty, best quality, extremely detailed, good light"
prompt = "A photo of street in chinese ancient style,chinese ink inpainting, white background, Song Dynasty, best quality, extremely detailed, good light"
# prompt = 'painting, people, tree, elephant'
# prompt = 'In the style of Chinese ink-and-wash painting, traditional Chinese realistic painting, Chinese ancient architecture, Night Scene in Chang an City, in Tang Dynasty'
# prompt = 'Night Scene in Chang an City, in Tang Dynasty'
# prompt = 'A young Chinese ancient male, in Tang Dynasty, tall and thin body type, very handsome, white official hat, white clothes with pattern, black boots.'
# prompt = 'An ancient Chinese boy, in Tang Dynasty, handsome, brown and beige clothes, white boots.'
# prompt = 'Some dragons are circling, chinese cartoon style, Tang Dynasty, perfect, extremely detailed'

# negative_prompt = 'yellow ground, gray ground'
# negative_prompt = 'people, person, bad, blur'
negative_prompt = 'blur, low quality'

lora_trigger = 'a photo in chinese cartoon style, '
# lora_trigger = 'liujiyou, Chinese ink painting, '
# lora_trigger = 'chinese peking opera '
# lora_trigger = 'QIEMANCN, '
# lora_trigger = 'QIEM6NCN, '
# lora_trigger = 'changshiban, '
# lora_trigger = 'a photo of chinese ancient drawing,'
# lora_trigger = 'a photo of a man in cartoon style,'
# lora_trigger = 'a photo of a boy in cartoon style,'
prompt = lora_trigger + prompt

generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

# get canny image
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/control/5_crop.jpg')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/scene/scene_content/场一 夜 外 长安城全景.png')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/donghuatest/content/libai.png')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/donghuatest/content/xiaohai.png')
image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/changeface/faces/liuwenran.jpg')

# image = image.resize((832, 1280))
# image = image.resize((1920, 960))
# image = image.resize((960, 1920))
# image = image.resize((1024, 1024))
image = image.resize((800, 1024))

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# folder_path = 'results/t2i_sd_xl_lora_canny/lora-trained-xl-leyuan-e4-checkpoint-300_xiaohai'
folder_path = 'results/test_liuwenran'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(10):
    image = pipeline(prompt, negative_prompt=negative_prompt,controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=canny_image).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))
