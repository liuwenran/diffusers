import torch
from diffusers import StableDiffusionPipeline
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

# added_prompt = 'painterly style, flat colours, illustration, bright and colourful, high contrast, Mythology, cinematic, detailed, atmospheric, 8k, corona render'
# added_prompt = 'blue color, illustration, high contrast, bright, high contrast, best quality, masterpiece, high res, realistic, cinematic, detailed, atmospheric, 8k, corona render'
added_prompt = 'bright and colourful, flat colours, best quality, masterpiece, high res, highly detailed'
negative_prompt = 'Watermark, Text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet, abnormal fingers, flowers easynegative'
negative_prompt = 'mutated hands, (poorly drawn hands:1.331),(fused fingers:1.61051), (too many fingers:1.61051), bad hands, missing fingers, extra digit'
negative_prompt = '(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, skin spots, acnes, skin blemishes, age spot, glans, (watermark:2),'

# ckpt_path = '/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5'
# ckpt_path = '/nvme/liuwenran/models/huggingface/mo-di-diffusion'
# ckpt_path = '/nvme/liuwenran/repos/diffusers/resources/civitai_ckpts/brav5'
# ckpt_path = 'SG161222/Realistic_Vision_V2.0'
# ckpt_path = 'resources/civitai_ckpts/guofeng33'
ckpt_path = 'resources/civitai_ckpts/chilloutmix_NiPrunedFp32Fix'

pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)

# lora_path = '/nvme/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/LowRA.safetensors'
# trigger = 'dark theme'
# lora_path = '/nvme/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/ConstructionyardAIV3.safetensors'
# trigger = 'constructionyardai'
# lora_path = '/nvme/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/breastinclassBetter.safetensors'
# trigger = '<lora:breastinclassbetter_v141:0.5>'
lora_path = '/mnt/petrelfs/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/Moxin_10.safetensors'
trigger = 'shuimobysim, wuchangshuo'
weight = 0.3
pipe = load_lora_weights(pipe, lora_path, weight, 'cuda', torch.float32)
# trigger = ''
lora_path = '/mnt/petrelfs/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/shukezouma_v1_1.safetensors'
trigger = trigger + ', shukezouma'
weight = 0.4
pipe = load_lora_weights(pipe, lora_path, weight, 'cuda', torch.float32)


pipe.load_textual_inversion("resources/civitai_textual_inversion/easynegative.safetensors")
trigger = trigger + ', easynegative'


# prompt_path = '/nvme/liuwenran/datasets/prompts/english_prompt.txt'
# prompt_lines = open(prompt_path).readlines()
prompt_base = "a handsome man, painting in water colours"
for ind, line in enumerate(range(10)):
    set_seed(ind)
    prompt = prompt_base + ', ' + trigger
    prompt += ', ' + added_prompt
    print(prompt)
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    image.save("results/chilloutmix_moxin_shukezouma_shuimo_man_colorful/" + "sd_" + '{:03d}'.format(ind) + ".png")

# prompt_path = '/nvme/liuwenran/datasets/prompts/english_prompt.txt'
# prompt_lines = open(prompt_path).readlines()

# for ind, line in enumerate(prompt_lines):
#     prompt = line.strip()
#     prompt += ', ' + trigger
#     prompt += added_prompt
#     print(prompt)
#     image = pipe(prompt, negative_prompt=negative_prompt).images[0]
#     image.save("results/sd_lora_complex_yardai/" + "sd_lora_" + '{:03d}'.format(ind) + ".png")
