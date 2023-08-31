import PIL
from PIL import Image
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.training_utils import set_seed
set_seed(20231)
generator = torch.Generator(device='cuda').manual_seed(666)


# model_id = "timbrooks/instruct-pix2pix"
model_id = "resources/instruct-pix2pix"
# model_id = "ckpts/sd-instructp2p-expression-five"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
# def download_image(url):
#     image = PIL.Image.open(requests.get(url, stream=True).raw)
#     image = PIL.ImageOps.exif_transpose(image)
#     image = image.convert("RGB")
#     return image
# image = download_image(url)

# img = '/nvme/liuwenran/datasets/rage-comic/yaoming-images/7.jpeg'
# img = '/nvme/liuwenran/datasets/chenkai/6.png'
# img = '/nvme/liuwenran/datasets/expression/1.png'
# img = '/nvme/liuwenran/datasets/gaoqiqiang/1.jpeg'
img = '/nvme/liuwenran/datasets/forvideo/Picture3.png'
image = Image.open(img)
image = image.resize((256, 256)).convert('RGB')

prompt = "add mountain to the background"

for i in range(10):
    edit = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=i * 0.1, guidance_scale=7, generator=generator)
    edit.images[0].save("results/instructp2p-caixunkun/greenhat_" + str(i) +".png")