from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

img_dir = '/nvme/liuwenran/datasets/zhou_zenmela_frames_resized/0001.jpg'

image = load_image(img_dir)
image = hed(image)


prompt = 'a girl, black hair, smoking' 
# prompt = 'a handsome man, silver hair, gray coat, smiling face, play basketball, in the city'

a_prompt = 'best quality, extremely detailed'
prompt = prompt + ', ' + a_prompt
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'



controlnet = ControlNetModel.from_pretrained(
    "resources/sd-controlnet-hed", torch_dtype=torch.float32
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Linaqruf/anything-v3.0", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float32
)

# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload()

save_folder = 'results/controlnet/anythingv3_hed_temp'
# save_folder = 'results/anythingv3_canny_caixunkun_dancing_begin_fps30_temp'
# save_folder = 'results/any3_canny_normal_zhou_zenmela_originimg2img_dividesteps_redhair'
import os
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# frame_file = '/nvme/liuwenran/datasets/caixukun_dancing_begin_fps30_frames/frames.txt'
frame_file = '/nvme/liuwenran/datasets/zhou_zenmela_fps10_frames/frames.txt'
lines = open(frame_file, 'r').readlines()
latent_image = None
last_latent_image = None
first_latent_image = None

init_noise_shape = (1, 4,  64, 64)
init_noise_all_frame = torch.randn(init_noise_shape).cuda()

for ind in range(len(lines)):
    img_dir = lines[ind].strip() 
    image = load_image(img_dir)
    image = hed(image)

    # if ind == 0:
    # latent_image = load_image(img_dir)
    # latent_image = latent_image.resize((image_resolution, image_resolution))
    # latent_image = None

    result = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=20).images[0]
    
    save_name = os.path.join(save_folder, 'res_{:0>4d}.jpg'.format(ind))

    result.save(save_name)
