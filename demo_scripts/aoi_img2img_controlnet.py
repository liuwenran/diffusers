from PIL import Image
import torch
import os
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
import numpy as np
from diffusers.training_utils import set_seed
import cv2
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel, FluxControlNetImg2ImgPipeline

set_seed(20231)
device = "cuda"

style_type = ['cartoon', 'oil painting', 'graffiti', 'flux cartoon'][3]
lora_path = None
lora_trigger = ''
from_single_file = False
base_model = ['sdxl', 'flux'][1]

if base_model == 'sdxl':
    weight_dtype = torch.float16
    controlnet_conditioning_scale = 0.6
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=weight_dtype
    )
else:
    controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)

# load the pipeline
if style_type == 'cartoon':
    ckpt_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--gsdf--CounterfeitXL/snapshots/4708675873bd09833aabc3fd4cb2de5fcd1726ac'
elif style_type == 'oil painting':
    ckpt_path = '/mnt/petrelfs/liuwenran/models/civitai/ckpts/protovisionXLHighFidelity3D_release0630Bakedvae.safetensors'
    lora_path = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/ClassipeintXL1.9.safetensors'
    lora_trigger = 'oil painting,'
    from_single_file = True
elif style_type == 'graffiti':
    ckpt_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--gsdf--CounterfeitXL/snapshots/4708675873bd09833aabc3fd4cb2de5fcd1726ac'
    lora_path = '/mnt/petrelfs/liuwenran/models/civitai/loras/sdxl_loras/Graffiti Comic.safetensors'
    lora_trigger = 'Graffiti Comic,'
elif style_type == 'flux cartoon':
    lora_path = '/mnt/petrelfs/liuwenran/models/civitai/loras/flux/flat_colour_anime_style_v3.4.safetensors'
    lora_trigger = 'Flat colour anime style image showing, '
else:
    raise ValueError(f"style_type {style_type} not supported")

if base_model == 'sdxl':
    if from_single_file:
        pipe_style = StableDiffusionXLImg2ImgPipeline.from_single_file(ckpt_path, torch_dtype=weight_dtype).to(device)
    else:
        pipe_style = StableDiffusionXLImg2ImgPipeline.from_pretrained(ckpt_path, torch_dtype=weight_dtype).to(device)
    if lora_path is not None:
        pipe_style.load_lora_weights(lora_path)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline(pipe_style.vae, pipe_style.text_encoder, pipe_style.text_encoder_2,
                                    pipe_style.tokenizer, pipe_style.tokenizer_2, pipe_style.unet, controlnet, pipe_style.scheduler)
else:
    pipe_style = FluxControlNetPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16).to(device)
    if lora_path is not None:
        pipe_style.load_lora_weights(lora_path)
    pipe = FluxControlNetImg2ImgPipeline(vae=pipe_style.vae, text_encoder=pipe_style.text_encoder, text_encoder_2=pipe_style.text_encoder_2,
                                    tokenizer=pipe_style.tokenizer, tokenizer_2=pipe_style.tokenizer_2, transformer=pipe_style.transformer, scheduler=pipe_style.scheduler, controlnet=pipe_style.controlnet)
    

pipe = pipe.to(device)

img = '/mnt/petrelfs/liuwenran/datasets/petdata/content/cat/cat512.jpg'
# img = '/mnt/petrelfs/liuwenran/datasets/petdata/content/cat/loki.jpg'
init_image = Image.open(img)
init_image = init_image.resize((1024, 1024)).convert('RGB')

prompt = "a cute cat in cartoon style, best quality, perfect, extremely detailed, 8k"
negative_prompt = "ugly, bad, disfigured, blur"

prompt = lora_trigger + prompt

image = np.array(init_image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

for controlnet_conditioning_scale in np.arange(0.5, 1, 0.1):
    for strength in np.arange(0.5, 0.9, 0.1):
        folder_path = f"results/aoi/{img.split('/')[-1].split('.')[0]}_{style_type}_controlnet{controlnet_conditioning_scale:.1f}-{strength:.2f}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i in range(2):
            if base_model == 'sdxl':
                image = pipe(prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=controlnet_conditioning_scale,
                                            image=init_image, control_image=canny_image, strength=strength).images[0]
            else:
                image = pipe(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale,
                            image=init_image, control_image=canny_image, strength=strength).images[0]
            image.save(os.path.join(folder_path, f"{i}.png"))