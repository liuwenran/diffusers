# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image

MIN_IND = 0
MAX_IND = 12
print(f'MIN {MIN_IND} MAX {MAX_IND}')

# initialize the models and pipeline
weight_dtype = torch.float16

controlnet_conditioning_scale = 0.6  # recommended for good generalization
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
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
)


# trained_ckpt = 'work_dirs/t2i-changshiban/t2i-changshiban-trainckpt-e5-fp16'
# pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
#     trained_ckpt, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
# )

pipeline.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors")

lora_dir = 'work_dirs/cctv/qianqiushisong/lora-trained-xl-qianqiuhuman-e4/checkpoint-300'
# lora_dir = 'work_dirs/t2i-changshiban/t2i-xiangsi-e4-fp16/checkpoint-600'
# lora_dir = 'work_dirs/t2i-changshiban/t2i-changshiban-fullsize720-e4/checkpoint-3400'
pipeline.load_lora_weights(lora_dir)
pipeline = pipeline.to("cuda")

# prompt
lora_trigger = 'a photo in cartoon style, '
# lora_trigger = 'changshiban,'


# role
# role = '/mnt/petrelfs/liuwenran/datasets/角色视图/dongtinglan.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/角色视图/images.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/new_characters.txt'
role = '/mnt/petrelfs/liuwenran/datasets/cctv/dongtinglan/sixpose.txt'

# prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/角色视图/prompts.txt'
# prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/prompts.txt'
# prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/new_prompts.txt'
prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/dongtinglan/prompt.txt'

prompt_lines = open(prompt_file, 'r').readlines()
prompt_dict = {}
for line in prompt_lines:
    line = line.strip()
    line_role = line.split(':')[0]
    line_prompt = line.split(':')[1]
    prompt_dict[line_role] = line_prompt


lines = open(role, 'r').readlines()
for ind, line in enumerate(lines):
    if '正' in line:
        temp = lines[ind]
        lines[ind] = lines[0]
        lines[0] = temp

for picset in range(1, 5):
    for ind, line in enumerate(lines):
        print('image ind ' + str(ind))
        line = line.strip()
        print(line)
        role_name = line.split('/')[-2]
        img_name = role_name + '_' + line.split('/')[-1].split('.')[0]

        prompt = 'an old chinese ancient officer, silver beard, Tang Dynasty, white clothes, black offical hat, black boots'
        # prompt = None
        # for role in prompt_dict.keys():
        #     if role == role_name.split('6')[0].strip():
        #         prompt = prompt_dict[role]
        #         break

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

        folder_path = 'results/dongtinglan/lora-trained-xl-qianqiuhuman-e4-checkpoint-300_origin'
        folder_path = os.path.join(folder_path, f'attn_set{picset}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        generator = torch.Generator(device=torch.device('cuda')).manual_seed(picset)
        if ind == 0:
            image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator, num_inference_steps=25, image=canny_image).images[0]
            front_image = image
        else:
            image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator, num_inference_steps=25, image=canny_image, ref_image=front_image).images[0]
        image.save(os.path.join(folder_path, str(ind) + ".png"))
        pipeline.unet.clear_bank()



