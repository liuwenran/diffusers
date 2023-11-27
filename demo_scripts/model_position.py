from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
import torch


weight_dtype = torch.float16

controlnet = ControlNetModel.from_pretrained(
    '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--diffusers--controlnet-canny-sdxl-1.0/snapshots/eb115a19a10d14909256db740ed109532ab1483c', torch_dtype=weight_dtype
)

vae_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15'
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)

pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, vae=vae, controlnet=controlnet, torch_dtype=weight_dtype
)
