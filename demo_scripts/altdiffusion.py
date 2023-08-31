import torch
from diffusers import AltDiffusionPipeline
pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m18", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
# "dark elf princess, highly detailed, d & d, fantasy, highly detailed, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and fuji choko and viktoria gavrilenko and hoang lap"
prompt = "黑暗精灵公主，非常详细，幻想，非常详细，数字绘画，概念艺术，敏锐的焦点，插图"
image = pipe(prompt).images[0]

image.save("resources/output/altdiffusion_m18.png")
