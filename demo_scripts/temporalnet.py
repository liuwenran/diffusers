from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image


hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/man.png")


image = hed(image)

controlnet = ControlNetModel.from_pretrained(
    "resources/sd-controlnet-hed", torch_dtype=torch.float32
)


file_path = "/nvme/liuwenran/models/huggingface/TemporalNet/diff_control_sd15_temporalnet_fp16.pth"
loaded = torch.load(file_path)

import ipdb;ipdb.set_trace();

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "resources/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float32
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload()

image = pipe("oil painting of handsome old man, masterpiece", image, num_inference_steps=20).images[0]

image.save('results/controlnet_hed/man_hed_out_fp16.png')
