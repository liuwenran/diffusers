import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel

from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image
import argparse

from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--min", type=int)
parser.add_argument("--max", type=int)
parser.add_argument("--type", type=int, default=0)
args = parser.parse_args()
MIN_IND = args.min
MAX_IND = args.max

controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
controlnet_model_xlab = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--XLabs-AI--flux-controlnet-canny-v3/snapshots/89efcb0f8e15c8d0d0912d93a4ff313e20db5eb6/flux-canny-controlnet-v3.safetensors'
controlnet_model_xlab_ckpt = load_file(controlnet_model_xlab)
controlnet.load_state_dict(controlnet_model_xlab_ckpt, strict=False)


pipeline = FluxControlNetPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16)

# lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/qianqiushisong_s3/flux_shuimo_lora/checkpoint-5000/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/qianqiushisong_s3/flux_shouhui_lora_e3/pytorch_lora_weights.safetensors'
lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/demo_scripts/flat_colour_anime_style_v3.4.safetensors'
pipeline.load_lora_weights(lora_dir)


# lora_trigger = 'a photo in chinese cartoon style, characters in animations, '
lora_trigger = 'Flat colour anime style image showing, '

pipeline = pipeline.to("cuda")
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)


if args.type == 0:
    # role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s3/content/国风手绘风格——悯农/roles.txt'
    role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s3/content/国风水墨仙侠风格——望岳/roles_front.txt'
else:
    # role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/zaofabaidicheng_role.txt'
    # role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/zaofabaidicheng_role_rework.txt'
    # role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/zaofabaidicheng_role_rework2.txt'
    # role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/zaofabaidicheng_role_rework3.txt'
    role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/zaofabaidicheng_role_rework4.txt'

# prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/角色视图/prompts.txt'
# prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/prompts.txt'
# prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/new_prompts.txt'

if args.type == 0:
    prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s3/content/国风水墨仙侠风格——望岳/wangyue.txt'
    # prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s3/content/仙侠风格——望庐山瀑布/wanglushan.txt'
    # prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s3/content/传统二维风格——九月九日忆山东兄弟/yishandong.txt'
else:
    prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/zaofabaidicheng.txt'


prompt_lines = open(prompt_file, 'r').readlines()
prompt_dict = {}
for line in prompt_lines:
    line = line.strip()
    line_role = line.split(':')[0]
    line_prompt = line.split(':')[1]
    prompt_dict[line_role] = line_prompt

if args.type == 0:
    # output_root_path = 'results/cctv/qianqiushisong_s3/style_shuimo_wangyue'
    # output_root_path = 'results/cctv/qianqiushisong_s3/style_shouhui_minnong'
    # output_root_path = 'results/cctv/qianqiushisong_s3/style_xianxia_wanglushan'
    # output_root_path = 'results/cctv/qianqiushisong_s3/style_chuantong_yishandong'
    # output_root_path = 'results/cctv/qianqiushisong_s3/style_shuimo_wangyue_flux06'
    output_root_path = 'results/cctv/qianqiushisong_s3/flux_flat_colour_anime_style_v3.4'
else:
    output_root_path = 'results/cctv/qianqiushisong_s2e2_rework_4'

lines = open(role_file, 'r').readlines()

for ind, line in enumerate(lines):
    if ind >= MIN_IND and ind < MAX_IND:
        print('image ind ' + str(ind) + f' in {len(lines)}' + f' type {args.type}')
        line = line.strip()
        print(line)
        role_name = line.split('/')[-2]
        img_name = line.split('/')[-1].split('.')[0]
        # role_name = img_name

        # prompt = 'an old chinese man'
        # prompt = 'an old chinese officer, silver beard, white clothes, black offical hat, black boots'
        prompt = None
        for role in prompt_dict.keys():
            if role == role_name:
                prompt = prompt_dict[role]
                break
        # prompt = ', perfect, best quality, masterpeice'

        if '背' in line:
            prompt = prompt + ',back view, '
        elif '左' in line or '右' in line:
            prompt = prompt + ',side view,'
        elif '正' in line:
            prompt = prompt + ',front view,'
        prompt = lora_trigger + prompt
        print(prompt)

        image = load_image(line)

        # human img
        image = image.resize((1080, 1920))
        
        # tools img
        # image_size = image.size
        # if image_size[0] < 512 or image_size[1] < 512:
        #     new_size = (image_size[0] * 2, image_size[1] * 2)
        #     image = image.resize(new_size)
        # elif image_size[0] > 2048 or image_size[1] > 2048:
        #     new_size = (int(image_size[0] / 2), int(image_size[1] / 2))
        #     image = image.resize(new_size)

        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        # Define the kernel for dilation. You can change the size to adjust the thickness of the edges.
        kernel = np.ones((3,3), np.uint8)
        # Dilation
        image = cv2.dilate(image, kernel, iterations=1)

        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        folder_path = os.path.join(output_root_path, f'{role_name}_{img_name}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for ind in range(10):
            image = pipeline(prompt, width=1080, height=1920, controlnet_conditioning_scale=0.6, num_inference_steps=25, control_image=canny_image).images[0]
            image.save(os.path.join(folder_path, str(ind) + ".png"))
            break
