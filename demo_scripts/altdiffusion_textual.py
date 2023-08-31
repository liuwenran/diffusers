import torch
from diffusers import AltDiffusionPipeline
from diffusers.training_utils import set_seed


set_seed(1)

added_prompt = '绘画风格，单色，插图，明亮多彩，高对比度，神话，电影，细节，大气，8k，电晕渲染'
negative_prompt = '水印、文字、删节、变形、解剖不良、毁容、脸画得不好、变异、多余的肢体、丑陋、画得不好的手、缺肢、漂浮的肢体、断肢、断头、畸形的手、长脖子、变异的手和手指 , 坏手, 缺手指, 裁剪, 质量最差, 低质量, 突变, 画得不好, 巨大的小腿, 坏手, 融合的手, 缺手, 消失的手臂, 消失的大腿, 消失的小腿, 消失的腿, 消失的手指, 融合的手指, 眼睛比例异常，手异常，腿异常，脚异常，手指异常'

# added_prompt = 'painterly style, flat colours, illustration, bright and colourful, high contrast, Mythology, cinematic, detailed, atmospheric, 8k, corona render'
# negative_prompt = 'Watermark, Text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet, abnormal fingers, flowers easynegative'

# pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m18", torch_dtype=torch.float16)
pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)


# lora_path = '/nvme/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/LowRA.safetensors'
# # trigger = 'dark theme'
# lora_path = '/nvme/liuwenran/repos/sd-webui/stable-diffusion-webui/models/Lora/ConstructionyardAIV3.safetensors'
# trigger = 'constructionyardai'
# pipe = load_lora_weights(pipe, lora_path, 1.0, 'cuda', torch.float32)
pipe.load_textual_inversion("./resources/text_embedding/WarriorStyle.pt")
trigger = 'WarriorStyle'


prompt_path = '/nvme/liuwenran/datasets/prompts/chinese_prompt.txt'
prompt_lines = open(prompt_path).readlines()

for ind, line in enumerate(prompt_lines):
    prompt = line.strip()
    prompt += ', ' + added_prompt
    prompt = trigger + ', ' + prompt
    print(prompt)
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    # image = pipe(prompt).images[0]
    image.save("results/altdiffusion_textual/" + "original_ad_" + '{:03d}'.format(ind) + ".png")
