# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image


# initialize the models and pipeline
weight_dtype = torch.float16

controlnet_conditioning_scale = 1.0  # recommended for good generalization
# controlnet = ControlNetModel.from_pretrained(
#     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=weight_dtype
# )

controlnet = ControlNetModel.from_pretrained(
    '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--diffusers--controlnet-canny-sdxl-1.0/snapshots/eb115a19a10d14909256db740ed109532ab1483c', torch_dtype=weight_dtype
)


vae_path = "madebyollin/sdxl-vae-fp16-fix"
vae_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15'
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
# pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--gsdf--CounterfeitXL/snapshots/4708675873bd09833aabc3fd4cb2de5fcd1726ac'
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
)


# trained_ckpt = 'work_dirs/t2i-changshiban/t2i-changshiban-trainckpt-e5-fp16'
# pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
#     trained_ckpt, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
# )

# pipeline.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors")

# lora_dir = 'work_dirs/cctv/lora-trained-xl-willow-e4/checkpoint-300'
# lora_dir = 'work_dirs/t2i-changshiban/t2i-xiangsi-e4-fp16/checkpoint-600'
# lora_dir = 'work_dirs/t2i-changshiban/t2i-changshiban-fullsize720-e4/checkpoint-3400'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/qianqiushisong3/lora-trained-xl-weiqishaonian-character-v214-e4-300/checkpoint-300/pytorch_lora_weights.safetensors'
lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/qianqiushisong3/lora-trained-xl-weiqishaoniao_person-v214-e4-300/checkpoint-300/pytorch_lora_weights.safetensors'

pipeline.load_lora_weights(lora_dir)
pipeline = pipeline.to("cuda")

# prompt
# lora_trigger = 'a photo in cartoon style, '
lora_trigger = 'characters in chinese cartoon style, '
# lora_trigger = 'changshiban,'


# role
# role = '/mnt/petrelfs/liuwenran/datasets/角色视图/dongtinglan.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/角色视图/images.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/new_characters.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/dongtinglan/sixpose.txt'
origin_imgs_file = '/mnt/petrelfs/liuwenran/datasets/cctv/changeface/shierchangsheng_test_7.txt'


lines = open(origin_imgs_file, 'r').readlines()

for ind, line in enumerate(lines):
    print('image ind ' + str(ind))
    line = line.strip()

    prompt = ''
    prompt = lora_trigger + prompt
    print(prompt)

    image = load_image(line)
    image = image.resize((1080, 1920))

    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    folder_path = '/mnt/petrelfs/liuwenran/datasets/cctv/changeface/shierchangsheng_test_7_cartoon_noattn'
    img_name = line.split('/')[-1]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    generator = torch.Generator(device=torch.device('cuda')).manual_seed(1024)
    # if ind == 0:
    controlnet_conditioning_scale = 1.2
    image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator, num_inference_steps=25, image=canny_image).images[0]
    # else:
    #     controlnet_conditioning_scale = 1.2
    #     image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator, num_inference_steps=25, image=canny_image, ref_image=ref_image).images[0]
    # ref_image = image

    image.save(os.path.join(folder_path, img_name))
    # pipeline.unet.clear_bank()



