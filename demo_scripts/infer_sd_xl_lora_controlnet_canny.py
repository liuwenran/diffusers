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

controlnet_conditioning_scale = 0.8  # recommended for good generalization
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
# lora_dir = 'work_dirs/lora-trained-xl-xuanwu-e4/checkpoint-300'
# lora_dir = 'work_dirs/lora-trained-xl-ancientpic-e4/checkpoint-300'
# lora_dir = 'work_dirs/t2i-changshiban-e4-fp16/checkpoint-2700'
# lora_dir = 'work_dirs/t2i-changshiban/t2i-changshiban-fullsize720-e4/checkpoint-3400'
# lora_dir = 'work_dirs/cctv/lora-trained-xl-3dstyle-e4/checkpoint-300'
# lora_dir = 'work_dirs/cctv/lora-trained-xl-comic-e4/checkpoint-300'
# lora_dir = 'work_dirs/cctv/lora-trained-xl-leyuan-e4/checkpoint-300'
# lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/guhua/civitai_lora/20240202-1706856411592-0008.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/old_version/diffusers/work_dirs/cctv/guhua/lora-trained-xl-luoshenfu/checkpoint-300/pytorch_lora_weights.safetensors'
lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/loras/20240202-1706856411592-0008.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/loras/henv.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/loras/Dreamyvibes artstyle SDXL - Trigger with dreamyvibes artstyle.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/loras/国风插画SDXL.safetensors'
# lora_dir = '/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/loras/ghibli_last.safetensors'

if args.type == 0:
    pipeline.load_lora_weights(lora_dir)

pipeline = pipeline.to("cuda")

# prompt = "A photo of a chinese old man in cartoon style, Tang Dynasty, portrait, without hands, perfect, extremely detailed, 8k"
# prompt = "A photo of chinese buildings in cartoon style, chinese ancient, Song Dynasty, without people, best quality, extremely detailed, good light"
# prompt = "A photo of chinese buildings in cartoon style, chinese ancient, Song Dynasty, without people, best quality, extremely detailed, good light"
# prompt = "A photo of a man in chinese ancient style, chinese ink inpainting, Song Dynasty, without people, best quality, extremely detailed, good light"
# prompt = "A photo of a man in chinese ancient style, chinese ink inpainting, Song Dynasty, best quality, extremely detailed, good light"
# prompt = "A photo of a man in chinese ancient style, Tang Dynasty, best quality, extremely detailed, good light"
# prompt = "A photo of street in chinese ancient style,chinese ink inpainting, white background, Song Dynasty, best quality, extremely detailed, good light"
# prompt = 'painting, people, tree, elephant'
# prompt = 'In the style of Chinese ink-and-wash painting, traditional Chinese realistic painting, Chinese ancient architecture, Night Scene in Chang an City, in Tang Dynasty'
# prompt = 'Night Scene in Chang an City, in Tang Dynasty'
# prompt = 'A young Chinese ancient male, in Tang Dynasty, tall and thin body type, very handsome, white official hat, white clothes with pattern, black boots.'
# prompt = 'An ancient Chinese boy, in Tang Dynasty, handsome, brown and beige clothes, white boots.'
# prompt = 'Some dragons are circling, chinese cartoon style, Tang Dynasty, perfect, extremely detailed'
# prompt = 'an ancient chinese drawing, chinese cartoon style, Tang Dynasty, perfect, extremely detailed'
# prompt = 'outside, in the mountain, stones, chinese cartoon style, Tang Dynasty, perfect, extremely detailed'
# prompt = 'scenes in chinese cartoon style'
prompt = ' Props in animations, chinese cartoon style,'

# prompt_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/gongbi_prompt/prompt_tools.txt', 'r').read().splitlines()
# prompt_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/gongbi_prompt/prompt_scenes.txt', 'r').read().splitlines()
# prompt_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/tools_prompts.txt').read().splitlines()

# negative_prompt = 'yellow ground, gray ground'
# negative_prompt = 'people, person, bad, blur'
negative_prompt = 'blur, low quality'

lora_trigger = ''
# lora_trigger = 'a photo in chinese cartoon style, '
# lora_trigger = 'liujiyou, Chinese ink painting, '
# lora_trigger = 'chinese peking opera '
# lora_trigger = 'QIEMANCN, '
# lora_trigger = 'QIEM6NCN, '
# lora_trigger = 'changshiban, '
# lora_trigger = 'a photo of chinese ancient drawing,'
# lora_trigger = 'a photo of a man in cartoon style,'
# lora_trigger = 'a photo of a boy in cartoon style,'
# lora_trigger = 'CN_SP style, '
# lora_trigger = 'jzcg036, '
if args.type == 0:
    lora_trigger = 'chinese traditional minimalism, '
# lora_trigger = 'mn, '
# lora_trigger = 'Dreamyvibes Artstyle, '
# lora_trigger = 'guofeng, chinese style'
# lora_trigger = 'ghibli'



prompt = lora_trigger + prompt

generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

# get canny image
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/control/5_crop.jpg')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/scene/scene_content/场一 夜 外 长安城全景.png')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/donghuatest/content/libai.png')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/donghuatest/content/xiaohai.png')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/changeface/faces/liuwenran.jpg')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/luoshen/20240327-150425.png')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/luoshen/20240327-160003.png')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/luoshen/20240327-164917.jpg')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/luoshen/20240327-165209.jpg')
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/guhua/content/20240327-191918.png')
# image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/guhua/content_character/content_character.txt').read().splitlines()
# image = load_image('/mnt/petrelfs/liuwenran/datasets/cctv/guhua/style/20240327-190647.jpg')
# image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/driving_characters.txt').read().splitlines()
# image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/scene2.txt').read().splitlines()
# image_lines = ['/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/driving_characters/20240510-202552.jpg']
# image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/driving_scenes1.txt').read().splitlines()
# image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/driving_scenes2.txt').read().splitlines()
# image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/wyy_baizhangji/baizhangji.txt').read().splitlines()
# image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/tools_rework.txt').read().splitlines()
# image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/others/others.txt').read().splitlines()
image_lines = open('/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s2e1e2/tools2.txt').read().splitlines()

# folder_path = 'results/t2i_sd_xl_lora_canny/lora-trained-xl-leyuan-e4-checkpoint-300_xiaohai'
# root_path = 'results/cctv/qianqiushisong_s2e2_scene2_dreamvibes'
# root_path = 'results/cctv/qianqiushisong_s2e2_scene2_anythingxl'
# root_path = 'results/cctv/qianqiushisong_s2_e1_scenes'
# root_path = 'results/cctv/qianqiushisong_s2_e2_scenes_refined'
# root_path = 'results/cctv/qianqiushisong_s2e2_driving_characters_guofengxl'

# root_path = 'results/cctv_platform/gongbi_result_scenes_1880_test'
# root_path = 'results/cctv_platform/gongbi_result_baizhangji_1'
# root_path = 'results/cctv/qianqiushisong_s2_e1_tools_rework'
# root_path = 'results/cctv/qianqiushisong_s2_e2_others'
# root_path = 'results/cctv/qianqiushisong_s2_e1_others'
root_path = 'results/cctv/qianqiushisong_s2_e2_tools'

init_prompt = prompt
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
        prompt = init_prompt 

        print(f'ind {ind} prompt: {prompt}')
        for ind in range(10):        
            image = pipeline(prompt, negative_prompt=negative_prompt,controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=canny_image).images[0]
            image.save(os.path.join(folder_path, str(ind) + ".png"))

# prompt_lines = ['bamboo']

# for ind, line in enumerate(prompt_lines):
#     prompt = lora_trigger + ',' + line
#     print(f'prompt: {line}')

#     folder_path = os.path.join(root_path, str(ind))
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     for ind in range(4):
#         image = pipeline(prompt, width=1880, height=800, negative_prompt=negative_prompt,num_inference_steps=25).images[0]
#         image.save(os.path.join(folder_path, str(ind) + ".png"))
