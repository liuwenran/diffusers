import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel
controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16)

lora_dir = '/mnt/petrelfs/liuwenran/forks/diffusers/work_dirs/cctv/qianqiushisong_s3/flux_shuimo_lora/pytorch_lora_weights.safetensors'
pipe.load_lora_weights(lora_dir)

pipe.to("cuda")
control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
prompt = "a photo in chinese cartoon style, characters in animations, a girl "
image = pipe(prompt, control_image=control_image, controlnet_conditioning_scale=0.6, num_inference_steps=28, guidance_scale=3.5).images[0]
image.save("flux.png")