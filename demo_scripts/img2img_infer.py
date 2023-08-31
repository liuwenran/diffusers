import requests
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

from diffusers.training_utils import set_seed
set_seed(20231)

# load the pipeline
# ckpt_path = "ckpts/sd-dreambooth-textencoder-classprompt-gaoqiqiang"
ckpt_path = '/nvme/liuwenran/models/huggingface/mo-di-diffusion'

device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(ckpt_path).to(device)

# let's download an initial image
# img = '/nvme/liuwenran/datasets/chenkai/1.jpg'
img = '/nvme/liuwenran/datasets/forvideo/Picture3.png'
# img = '/nvme/liuwenran/datasets/forvideo/0b69cb03-37e9-43a8-900d-d02de2cb8213.jpeg'
# img = '/nvme/liuwenran/datasets/others/3.jpg'
# img = '/nvme/liuwenran/datasets/rage-comic/yaoming-images/2.jpeg'
init_image = Image.open(img)
init_image = init_image.resize((512, 512)).convert('RGB')

prompt = "a man wear a green hat"
# prompt = None

strength = 0.1
for i in range(10):
    images = pipe(prompt=prompt, image=init_image, strength=strength + i * 0.1, guidance_scale=7.5).images
    images[0].save("results/mo-di/caixukun-"+str(strength + i * 0.1)+".png")