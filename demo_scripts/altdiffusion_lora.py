import torch
from diffusers import AltDiffusionPipeline
from diffusers.training_utils import set_seed
from safetensors.torch import load_file
from collections import defaultdict


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            continue
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


set_seed(1)

added_prompt = '绘画风格，单色，插图，明亮多彩，高对比度，神话，电影，细节，大气，8k，电晕渲染'
negative_prompt = '水印、文字、删节、变形、解剖不良、毁容、脸画得不好、变异、多余的肢体、丑陋、画得不好的手、缺肢、漂浮的肢体、断肢、断头、畸形的手、长脖子、变异的手和手指 , 坏手, 缺手指, 裁剪, 质量最差, 低质量, 突变, 画得不好, 巨大的小腿, 坏手, 融合的手, 缺手, 消失的手臂, 消失的大腿, 消失的小腿, 消失的腿, 消失的手指, 融合的手指, 眼睛比例异常，手异常，腿异常，脚异常，手指异常'

# added_prompt = 'painterly style, flat colours, illustration, bright and colourful, high contrast, Mythology, cinematic, detailed, atmospheric, 8k, corona render'
# negative_prompt = 'Watermark, Text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet, abnormal fingers, flowers easynegative'

pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m18", torch_dtype=torch.float16)
# pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)


# lora_path = '/nvme/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/LowRA.safetensors'
# # trigger = 'dark theme'
# lora_path = '/nvme/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/ConstructionyardAIV3.safetensors'
# trigger = 'constructionyardai'
# pipe = load_lora_weights(pipe, lora_path, 1.0, 'cuda', torch.float32)


prompt_path = '/nvme/liuwenran/datasets/prompts/chinese_prompt.txt'
prompt_lines = open(prompt_path).readlines()

for ind, line in enumerate(prompt_lines):
    prompt = line.strip()
    # prompt += ',' + trigger
    # prompt += added_prompt
    print(prompt)
    # image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    image = pipe(prompt).images[0]
    image.save("results/altdiffusion_m18_lora_yardai/" + "original_ad_" + '{:03d}'.format(ind) + ".png")
