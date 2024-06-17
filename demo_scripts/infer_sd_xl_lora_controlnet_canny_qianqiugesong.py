# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image
import argparse
import diffusers

parser = argparse.ArgumentParser()
parser.add_argument("--min", type=int)
parser.add_argument("--max", type=int)
parser.add_argument("--type", type=int)
args = parser.parse_args()
MIN_IND = args.min
MAX_IND = args.max

print(f'diffusers version {diffusers.__version__}')
print(f'type {args.type} MIN {MIN_IND} MAX {MAX_IND}')

# initialize the models and pipeline
weight_dtype = torch.float16

controlnet_conditioning_scale = 1.0  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=weight_dtype
)

# controlnet = ControlNetModel.from_pretrained(
#     '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--diffusers--controlnet-canny-sdxl-1.0/snapshots/eb115a19a10d14909256db740ed109532ab1483c', torch_dtype=weight_dtype
# )

vae_path = "madebyollin/sdxl-vae-fp16-fix"
# vae_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15'
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)
# vae = AutoencoderKL.from_single_file('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1/ckpts/sdxl_vae.safetensors')

if args.type == 0:
# pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
# pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--gsdf--CounterfeitXL/snapshots/4708675873bd09833aabc3fd4cb2de5fcd1726ac'
    pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
    )

else:
# single_file_path = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1/ckpts/animagineXLV31_v31.safetensors'
    single_file_path = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/ckpts/AnythingXL_xl.safetensors'
    pipeline = StableDiffusionXLControlNetPipeline.from_single_file(single_file_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype)


# trained_ckpt = 'work_dirs/t2i-changshiban/t2i-changshiban-trainckpt-e5-fp16'
# pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
#     trained_ckpt, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
# )

# pipeline.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors")

# lora_dir = 'work_dirs/cctv/qianqiushisong/lora-trained-xl-qianqiuhuman-e4/checkpoint-300/pytorch_lora_weights.safetensors'
# lora_dir = 'work_dirs/cctv/qianqiushisong3/lora-trained-xl-dawangriji-children-e4/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/qianqiushisong3/lora-trained-xl-dawangriji-children-e4-600/checkpoint-600/pytorch_lora_weights.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/qianqiushisong3/lora-trained-xl-weiqishaonian-woman-e4/checkpoint-300/pytorch_lora_weights.safetensors'
# lora_dir = 'work_dirs/cctv/qianqiushisong3/lora-trained-xl-dawangriji-character-e4-600/checkpoint-600'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/qianqiushisong3/lora-trained-xl-weiqishaonian-character-v214-e4-300/checkpoint-300/pytorch_lora_weights.safetensors'

if args.type == 0:
    lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/loras/20240202-1706856411592-0008.safetensors'
    # lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1/loras/Makoto Shinkai Style.safetensors'
    # lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1/loras/SDXL1.0_Essenz-series-by-AI_Characters_Style_YourNameWeatheringWithYouSuzumeMakotoShinkai-v1.4.safetensors'
    # lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1/loras/Sadamoto Yoshiyuki_XL_V2.safetensors'
    # lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/loras/henv.safetensors'


    # lora_dir = 'work_dirs/t2i-changshiban/t2i-xiangsi-e4-fp16/checkpoint-600'
    # lora_dir = 'work_dirs/t2i-changshiban/t2i-changshiban-fullsize720-e4/checkpoint-3400'
    # lora_dir = 'work_dirs/cctv/qianqiushisong3/lora-trained-xl-weiqishaonian-e4/checkpoint-300/pytorch_lora_weights.safetensors'
    # lora_dir = 'work_dirs/cctv/qianqiushisong3/lora-trained-xl-weiqishaonian-e4/checkpoint-300/pytorch_lora_weights.safetensors'

    pipeline.load_lora_weights(lora_dir)

    # prompt
    # lora_trigger = 'characters in chinese cartoon style, '
    # lora_trigger = 'changshiban,'
    lora_trigger = 'chinese traditional minimalism, '
    # lora_trigger = 'mn, chinese gongbi painting'
# lora_trigger = 'Makoto Shinkai Style page, '
# lora_trigger = 'anime screencap in mnst artstyle, '
# lora_trigger = 'souryuu asuka langley'
else:
    lora_trigger = ''

pipeline = pipeline.to("cuda")
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

# role = '/mnt/petrelfs/liuwenran/datasets/角色视图/dongtinglan.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/角色视图/images.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong2/new_characters.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong3/roles/roles.txt'
# role = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1/driving_characters.txt'

if args.type == 0:
    # role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/xiaoerchuidiao_role.txt'
    role_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/xiaoerchuidiao_role_rework.txt'
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
    prompt_file = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/all_characters/xiaoerechuidiao.txt'
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
    output_root_path = 'results/cctv/qianqiushisong_s2e1_rework'
else:
    output_root_path = 'results/cctv/qianqiushisong_s2e2_rework_4'

lines = open(role_file, 'r').readlines()

for ind, line in enumerate(lines):
    if ind >= MIN_IND and ind < MAX_IND:
        print('image ind ' + str(ind) + f' in {len(lines)}' + f' type {args.type}')
        line = line.strip()
        print(line)
        role_name = line.split('/')[-1].split(' ')[0]
        img_name = line.split('/')[-1].split('.')[0]

        # prompt = 'an old chinese man'
        # prompt = 'an old chinese officer, silver beard, white clothes, black offical hat, black boots'
        prompt = None
        for role in prompt_dict.keys():
            if role == role_name:
                prompt = prompt_dict[role]
                break

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

        folder_path = os.path.join(output_root_path, img_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for ind in range(10):
            image = pipeline(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=canny_image).images[0]
            image.save(os.path.join(folder_path, str(ind) + ".png"))


