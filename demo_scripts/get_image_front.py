import numpy as np
import cv2
from PIL import Image
import torch
import gradio as gr
import time

from diffusers import StableDiffusionInpaintPipeline

# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from segment_anything import build_sam, SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="weights/sam_vit_h_4b8939.pth").to(device))
print('load segement anything model.')


import torch

global origin_image_path
origin_image_path = None
global incremental_mask
incremental_mask = None

global count
count = 0

IMAGE_SIZE_THRE = 2000


def crop_image_pillow(img, divide=8):
    max_thre = IMAGE_SIZE_THRE
    width, height = img.size
    if width > max_thre:
        img = img.resize((max_thre, int(max_thre / width * height)))
        width, height = img.size
    if height > max_thre:
        img = img.resize((int(max_thre / height * width), max_thre))
        width, height = img.size

    left = width - width // divide * divide
    top = height - height // divide * divide
    right = width
    bottom = height

    img = img.crop((left, top, right, bottom))

    return img


def crop_image(img, divide=8):
    max_thre = IMAGE_SIZE_THRE
    height, width, _ = img.shape
    if width > max_thre:
        img = cv2.resize(img, (max_thre, int(max_thre / width * height)))
        height, width, _ = img.shape
    if height > max_thre:
        img = cv2.resize(img, (int(max_thre / height * width), max_thre))
        height, width, _ = img.shape

    top = height - height // divide * divide
    left = width - width // divide * divide
    img = img[top:, left:, :]
    return img


def crop_image_grey(img, divide=8):
    max_thre = IMAGE_SIZE_THRE
    height, width = img.shape
    if width > max_thre:
        img = cv2.resize(img, (max_thre, int(max_thre / width * height)))
        height, width = img.shape
    if height > max_thre:
        img = cv2.resize(img, (int(max_thre / height * width), max_thre))
        height, width = img.shape

    top = height - height // divide * divide
    left = width - width // divide * divide
    img = img[top:, left:]
    return img


def preview(image_path, draw_mask_path, use_drawed_mask=False):

    global origin_image_path
    if origin_image_path is None:
        origin_image_path = image_path

    draw_mask = cv2.imread(draw_mask_path)
    draw_mask = crop_image(draw_mask)
    gray = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
    draw_mask = gray > 0
    draw_mask = draw_mask.astype('uint8')

    if not use_drawed_mask:
        image = cv2.imread(image_path)
        image = crop_image(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.time()
        masks = mask_generator.generate(image)
        end = time.time()
        print('used time ' + str(end - start))

        indices = []
        for i, mask in enumerate(masks):
            bitwise_res = cv2.bitwise_and(mask['segmentation'].astype('uint8'), draw_mask)
            if np.sum(bitwise_res) > 0:
                indices.append(i)

        for seg_idx in indices:
            draw_mask = cv2.bitwise_or(masks[seg_idx]["segmentation"].astype('uint8'), draw_mask)

    global incremental_mask
    if incremental_mask is None:
        incremental_mask = draw_mask
    else:
        incremental_mask = cv2.bitwise_or(incremental_mask, draw_mask)

    output_draw_mask = incremental_mask * 255
    output_draw_mask = np.expand_dims(output_draw_mask, axis=2)
    output_draw_mask = np.repeat(output_draw_mask, repeats=3, axis=2)
    output_draw_mask = Image.fromarray(output_draw_mask)

    mask_image_binary = 1 - incremental_mask
    nb = np.expand_dims(mask_image_binary, axis=2)
    nm = np.repeat(nb, repeats=3, axis=2)

    image = cv2.imread(image_path)
    image = crop_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_image_masked = image * nm
    original_image_masked = Image.fromarray(original_image_masked)

    original_image_masked_front = image * (1 - nm)
    original_image_masked_front = Image.fromarray(original_image_masked_front)

    return original_image_masked_front, output_draw_mask, original_image_masked_front


draw_mask_path = '/mnt/petrelfs/liuwenran/forks/diffusers/results/output/out_draw_mask.png'

image = '/mnt/petrelfs/liuwenran/forks/diffusers/results/qianxiugesong/lora-trained-xl-qianqiuhuman-e4-0.6-fp16/士兵2 左45/0.png'

original_image_masked_front, output_draw_mask, original_image_masked_front = preview(image, draw_mask_path)

import ipdb;ipdb.set_trace();

a = 0