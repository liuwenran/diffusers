from diffusers import StableDiffusionPipeline

model_path = "ckpts/sd-dreambooth-gaoqiqiang"
pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe.to("cuda")

image = pipe(prompt="a cat is wearing a red hat").images[0]
image.save("qitan_pig.png")