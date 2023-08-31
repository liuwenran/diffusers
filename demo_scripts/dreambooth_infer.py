from diffusers import StableDiffusionPipeline
import torch

model_id = "ckpts/sd-dreambooth-songdaitaoci"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)

prompt = "a photo of songdaitaoci on the table"

for i in range(6):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save("results/dreambooth-songdaitaoci/songdaitaoci_ontable_" + str(i) + ".png")