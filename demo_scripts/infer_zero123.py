import os
import torch
import sys

path_to_add = "/mnt/petrelfs/liuwenran/forks/diffusers/examples/community"
sys.path.append(path_to_add)

from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from diffusers.utils import load_image

model_id = "kxic/zero123-xl" # zero123-105000, zero123-165000, zero123-xl

pipe = Zero1to3StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

num_images_per_prompt = 4

# test inference pipeline
# x y z, Polar angle (vertical rotation in degrees) 	Azimuth angle (horizontal rotation in degrees) 	Zoom (relative distance from center)
query_pose1 = [-75.0, 100.0, 0.0]
query_pose2 = [-20.0, 125.0, 0.0]
query_pose3 = [-55.0, 90.0, 0.0]

# load image
# H, W = (256, 256) # H, W = (512, 512)   # zero123 training is 256,256

# for batch input
input_image1 = load_image("https://cvlab-zero123-live.hf.space/file=/home/user/app/configs/4_blackarm.png")
input_image2 = load_image("https://cvlab-zero123-live.hf.space/file=/home/user/app/configs/8_motor.png")
input_image3 = load_image("https://cvlab-zero123-live.hf.space/file=/home/user/app/configs/7_london.png")
input_images = [input_image1, input_image2, input_image3]
query_poses = [query_pose1, query_pose2, query_pose3]

# # for single input
# H, W = (256, 256)
# input_images = [input_image2.resize((H, W), PIL.Image.NEAREST)]
# query_poses = [query_pose2]


# better do preprocessing
import cv2
import PIL
import time
from lovely_numpy import lo
from carvekit.api.high import HiInterface

def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def create_carvekit_interface():
    # Check doc strings for more information
    interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)

    return interface

def load_and_preprocess(interface, input_im):
    '''
    :param input_im (PIL Image).
    :return image (H, W, 3) array in [0, 1].
    '''
    # See https://github.com/Ir1d/image-background-remove-tool
    image = input_im.convert('RGB')

    image_without_background = interface([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, :, -1].astype(np.bool_)
    image[~foreground] = [255., 255., 255.]
    x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    image = image[y:y + h, x:x + w, :]
    image = PIL.Image.fromarray(np.array(image))

    # resize image such that long edge is 512
    image.thumbnail([200, 200], Image.Resampling.LANCZOS)
    image = add_margin(image, (255, 255, 255), size=256)
    image = np.array(image)

    return image

def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im

import numpy as np
import PIL.Image as Image

pre_images = []
models = dict()
print('Instantiating Carvekit HiInterface...')
models['carvekit'] = create_carvekit_interface()
if not isinstance(input_images, list):
    input_images = [input_images]
for raw_im in input_images:
    input_im = preprocess_image(models, raw_im, True)
    H, W = input_im.shape[:2]
    pre_images.append(Image.fromarray((input_im * 255.0).astype(np.uint8)))
input_images = pre_images

# infer pipeline, in original zero123 num_inference_steps=76
images = pipe(input_imgs=input_images, prompt_imgs=input_images, poses=query_poses, height=H, width=W,
              guidance_scale=3.0, num_images_per_prompt=num_images_per_prompt, num_inference_steps=50).images


# save imgs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
bs = len(input_images)
i = 0
for obj in range(bs):
    for idx in range(num_images_per_prompt):
        images[i].save(os.path.join(log_dir,f"obj{obj}_{idx}.jpg"))
        i += 1