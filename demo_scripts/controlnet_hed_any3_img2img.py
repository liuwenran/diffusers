# flake8: noqa
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from diffusers import DiffusionPipeline
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import cv2
import numpy as np
import os

import sys
sys.path
sys.path.append('/nvme/liuwenran/repos/diffusers/examples/community')

from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline

from diffusers.training_utils import set_seed

import torchvision.transforms as T
transform = T.ToPILImage()

set_seed(1)


hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

img_dir = '/nvme/liuwenran/datasets/zhou_zenmela_frames_resized/0001.jpg'
image = load_image(img_dir)
control_image = hed(image)


prompt = 'a girl, black hair, T-shirt, smoking, in the city' 
# prompt = 'a handsome man, silver hair, smiling, play basketball'
# prompt = 'a handsome man, red T-shirt, waving hands'

a_prompt = 'best quality, extremely detailed'
prompt = prompt + ', ' + a_prompt
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

# realistic_prompt = '(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3'
# prompt = prompt + ',' + realistic_prompt


# controlnet = [ ControlNetModel.from_pretrained("resources/sd-controlnet-hed", torch_dtype=torch.float32),
#             ControlNetModel.from_pretrained("resources/sd-controlnet-canny", torch_dtype=torch.float32) ]
controlnet = ControlNetModel.from_pretrained("resources/sd-controlnet-hed", torch_dtype=torch.float32)

# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "resources/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float32
# ).to("cuda")

# pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
#     "/nvme/liuwenran/.cache/huggingface/hub/models--Linaqruf--anything-v3.0/snapshots/1cfddae5cd94af6dbea2d4bb8dcaea8b7d215a1f", controlnet=controlnet, safety_checker=None).to("cuda")


# pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
#     "SG161222/Realistic_Vision_V1.4", controlnet=controlnet, safety_checker=None).to("cuda")


# pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
#     "Lykon/DreamShaper", controlnet=controlnet, safety_checker=None).to("cuda")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "WarriorMama777/AbyssOrangeMix2", controlnet=controlnet, safety_checker=None).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload()

debug_folder = 'results/controlnet_hed/result_debug'
save_folder = 'results/controlnet_hed/orangemix_hed_img2img_cat_unipc_w0.7'
# save_folder = 'results/controlnet_hed/any3_gen-1_dancer_input'
# save_folder = 'results/controlnet_hed/zhou_woyangni_fps10_frames_cat_unipc_w0.5_fireball256'
# save_folder = 'results/anythingv3_canny_caixunkun_dancing_begin_fps30_temp'
# save_folder = 'results/any3_canny_normal_zhou_zenmela_originimg2img_dividesteps_redhair'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# save_folder_second_loop = 'results/controlnet_hed/anythingv3_hed_img2img_cat_unipc_w0.5_second_loop'
# if not os.path.exists(save_folder_second_loop):
#     os.makedirs(save_folder_second_loop)

# frame_file = '/nvme/liuwenran/datasets/caixukun_dancing_begin_fps10_frames/frames.txt'
# frame_file = '/nvme/liuwenran/datasets/zhou_woyangni_fps10_frames_resized/frames.txt'
frame_file = '/nvme/liuwenran/datasets/zhou_zenmela_fps10_frames/frames.txt'
# frame_file = '/nvme/liuwenran/datasets/gen1/videos/dancer_input_fps10_frames_resized/frames.txt'
lines = open(frame_file, 'r').readlines()
latent_image = None
last_latent_image = None
first_latent_image = None
image_width = 512
image_height = 512
controlnet_conditioning_scale = 0.7

init_noise_shape = (1, 4, image_height // 8, image_width // 8)
init_noise_all_frame = torch.randn(init_noise_shape).cuda()

init_noise_shape_cat = (1, 4, image_height // 8, image_width // 8 * 3)
init_noise_all_frame_cat = torch.randn(init_noise_shape_cat).cuda()

generator = None

all_input_images = []
all_hed_images = []
all_result_images = []

# first result
img_dir = lines[0].strip()
image = load_image(img_dir)
image = image.resize((image_width, image_height))
hed_image = hed(image, image_resolution=image_width)
result = pipe(
    controlnet_conditioning_image=hed_image,
    # image=condition_images,
    image=image, 
    prompt=prompt,
    negative_prompt=negative_prompt,
    strength=0.75,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    num_inference_steps=20,
    generator=generator,
    latents=init_noise_all_frame).images[0]
first_result = result
first_hed = hed_image
last_result = result
last_hed = hed_image

last_concat_hed = None
for ind in range(len(lines)):
    img_dir = lines[ind].strip() 
    image = load_image(img_dir)
    image = image.resize((image_width, image_height))
    hed_image = hed(image, image_resolution=image_width)

    all_input_images.append(image)
    all_hed_images.append(hed_image)

    # canny_image = np.array(image)
    # low_threshold = 100
    # high_threshold = 200
    # canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
    # canny_image = canny_image[:, :, None]
    # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    # canny_image = Image.fromarray(canny_image)

    # condition_images = [hed_image, canny_image]

    concat_img = Image.new("RGB", (image_width * 3, image_height))
    concat_img.paste(last_result, (0, 0))
    # img.paste(p.init_images[0], (initial_width, 0))
    concat_img.paste(image, (image_width, 0))
    concat_img.paste(first_result, (image_width * 2, 0))
    concat_img.save(os.path.join(debug_folder, 'concat_image_{:0>4d}.jpg'.format(ind)))
    # import ipdb;ipdb.set_trace();

    concat_hed = Image.new("RGB", (image_width * 3, image_height), "black")
    concat_hed.paste(last_hed, (0, 0))
    concat_hed.paste(hed_image, (image_width, 0))
    concat_hed.paste(first_hed, (image_width * 2, 0))
    concat_hed.save(os.path.join(debug_folder, 'concat_hed_{:0>4d}.jpg'.format(ind)))
    # import ipdb;ipdb.set_trace();

    # if ind == 0:
    #     result = pipe(
    #                     controlnet_conditioning_image=hed_image, 
    #                     # image=condition_images,
    #                     image=image, 
    #                     prompt=prompt,
    #                     negative_prompt=negative_prompt,
    #                     strength=0.75,
    #                     controlnet_conditioning_scale=controlnet_conditioning_scale,
    #                     num_inference_steps=20,
    #                     generator=generator,
    #                     latents=init_noise_all_frame).images[0]
    
    # else:
    # if last_concat_hed is not None:
    #     last_concat_hed = np.array(last_concat_hed).astype(np.float32)
    #     momentum = 0.5
    #     concat_hed = np.array(concat_hed).astype(np.float32)
    #     concat_hed = momentum * concat_hed + (1 - momentum) * last_concat_hed
    #     concat_hed = Image.fromarray(np.uint8(concat_hed))
    
    # last_concat_hed = concat_hed

    # concat_hed = transform(torch.permute(concat_hed, (2,0,1)))

    result = pipe(
        controlnet_conditioning_image=concat_hed,
        # image=condition_images,
        image=concat_img, 
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=0.75,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=20,
        generator=generator,
        latents=init_noise_all_frame_cat,
        save_control_frame_ind=ind,
        control_save_dir=save_folder.split('/')[-1]).images[0]
    result = result.crop(
        (image_width, 0, image_width * 2, image_height))
    
    all_result_images.append(result)
        
    # if ind == 0:
    #     first_result = result
    #     first_hed = hed_image
    last_result = result
    last_hed = hed_image
    
    save_name = os.path.join(save_folder, '{:0>4d}.jpg'.format(ind))

    result.save(save_name)

video_name = save_folder.split('/')[-1]
cmd = 'ffmpeg -r 10 -i ' + save_folder + '/%04d.jpg -b:v 30M -vf fps=10 results/controlnet_hed/' + video_name + '.mp4'
os.system(cmd)

# for ind in range(len(all_input_images)):
#     image = all_input_images[ind]
#     hed_image = all_hed_images[ind]


#     if ind > 0:
#         concat_img = Image.new("RGB", (image_width * 3, image_height))
#         concat_img.paste(last_result, (0, 0))
#         # img.paste(p.init_images[0], (initial_width, 0))
#         concat_img.paste(all_result_images[ind], (image_width, 0))
#         concat_img.paste(first_result, (image_width * 2, 0))
#         concat_img.save(os.path.join(debug_folder, 'concat_image_{:0>4d}.jpg'.format(ind)))
#         # import ipdb;ipdb.set_trace();


#         concat_hed = Image.new("RGB", (image_width * 3, image_height), "black")
#         concat_hed.paste(last_hed, (0, 0))
#         concat_hed.paste(hed_image, (image_width, 0))
#         concat_hed.paste(first_hed, (image_width * 2, 0))
#         concat_hed.save(os.path.join(debug_folder, 'concat_hed_{:0>4d}.jpg'.format(ind)))
#         # import ipdb;ipdb.set_trace();

#     if ind == 0:
#         result = pipe(
#                         controlnet_conditioning_image=hed_image, 
#                         # image=condition_images,
#                         image=image, 
#                         prompt=prompt,
#                         negative_prompt=negative_prompt,
#                         strength=0.75,
#                         controlnet_conditioning_scale=controlnet_conditioning_scale,
#                         num_inference_steps=20,
#                         generator=generator,
#                         latents=init_noise_all_frame).images[0]
    
#     else:
#         result = pipe(
#                         controlnet_conditioning_image=concat_hed, 
#                         # image=condition_images,
#                         image=concat_img, 
#                         prompt=prompt,
#                         negative_prompt=negative_prompt,
#                         strength=0.75,
#                         controlnet_conditioning_scale=controlnet_conditioning_scale,
#                         num_inference_steps=20,
#                         generator=generator,
#                         latents=init_noise_all_frame_cat).images[0]
#         result = result.crop(
#                     (image_width, 0, image_width * 2, image_height))
            
#     if ind == 0:
#         first_result = result
#         first_hed = hed_image
#     last_result = result
#     last_hed = hed_image
    
#     save_name = os.path.join(save_folder_second_loop, '{:0>4d}.jpg'.format(ind))

#     result.save(save_name)