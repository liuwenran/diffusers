# from diffusers import StableDiffusionImageVariationPipeline
# from PIL import Image
# from torchvision import transforms

# device = "cuda"
# sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
#     "lambdalabs/sd-image-variations-diffusers",
#     revision="v2.0")
# sd_pipe = sd_pipe.to(device)

# im = Image.open("/mnt/petrelfs/liuwenran/repos/HumanAnimation/data/infer_ref_img/blue_boy.jpg")
# tform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(
#         (224, 224),
#         interpolation=transforms.InterpolationMode.BICUBIC,
#         antialias=False,
#         ),
#     transforms.Normalize(
#       [0.48145466, 0.4578275, 0.40821073],
#       [0.26862954, 0.26130258, 0.27577711]),
# ])
# inp = tform(im).to(device)
# import ipdb;ipdb.set_trace();
# out = sd_pipe(inp, guidance_scale=3)
# out["images"][0].save("result.jpg")

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from io import BytesIO
import requests

pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
)
pipe = pipe.to("cuda")

# url = "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200"

# response = requests.get(url)
image = Image.open("/mnt/petrelfs/liuwenran/repos/HumanAnimation/data/infer_ref_img/blue_boy.jpg").convert("RGB")

for guidance_scale in range(2, 25):
    out = pipe(image, num_images_per_prompt=4, guidance_scale=guidance_scale)
    for res_ind in range(4):
        out["images"][res_ind].save(f"results/sd15_image_variation/guidance_scale_{guidance_scale}_result{res_ind}.jpg")