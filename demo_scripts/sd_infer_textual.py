import torch
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import set_seed

set_seed(1)

added_prompt = 'painterly style, flat colours, illustration, bright and colourful, high contrast, Mythology, cinematic, detailed, atmospheric, 8k, corona render'
negative_prompt = 'Watermark, Text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet, abnormal fingers, flowers easynegative'

ckpt_path = '/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5'
# ckpt_path = '/nvme/liuwenran/models/huggingface/mo-di-diffusion'

pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)

pipe.load_textual_inversion("./resources/text_embedding/WarriorStyle.pt")
trigger = 'WarriorStyle'

prompt_path = '/nvme/liuwenran/datasets/prompts/english_prompt.txt'
prompt_lines = open(prompt_path).readlines()

for ind, line in enumerate(prompt_lines):
    prompt = line.strip()
    prompt += ', ' + added_prompt
    prompt = trigger + ', ' + prompt
    print(prompt)
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    image.save("results/sd_textual_warrior/" + "sd_textual_" + '{:03d}'.format(ind) + ".png")
