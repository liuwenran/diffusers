import torch
import numpy as np
import os
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image


depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_model_cpu_offload()


def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


lora_dir = 'lora-trained-xl-fp16train3k/checkpoint-1200'
# lora_dir = 'lora-trained-xl-fp16train2k'
# pipe.load_lora_weights(lora_dir)

# prompt = "A photo of painting in sks style, elephant, with a baby elephant"
prompt = "A tiger, realistic, in the wild, oil painting, best quality, extremely detailed, good light"
# image = load_image('/mnt/petrelfs/liuwenran/datasets/shanhaibailing/control/giraffe.jpg')
# image = image.crop((0, 0, 1024, 1024))
# image = load_image('/mnt/petrelfs/liuwenran/datasets/shanhaibailing/control/elephant_blue.jpeg')
image = load_image('/mnt/petrelfs/liuwenran/datasets/shanhaibailing/control/tiger_seg.png')
image = image.resize((1024, 1024))
controlnet_conditioning_scale = 0.5  # recommended for good generalization

depth_image = get_depth_map(image)

folder_path = 'results/sd_xl_controlnet_depth/tiger_depth'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)
for ind in range(100):
    image = pipe(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=25, image=depth_image).images[0]
    image.save(os.path.join(folder_path, "elephant_shanhai_depth_" + str(ind) + ".png"))

# images = pipe(
#     prompt, image=depth_image, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale,
# ).images
# images[0]

# images[0].save(f"stormtrooper.png")
