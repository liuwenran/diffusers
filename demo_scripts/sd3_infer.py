import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

for i in range(10):
    image = pipe(
        "A beautiful chinese woman is dancing, perfect, 8k, extremely detailed, realistic, high quality, high resolution, high definition",
        negative_prompt="bad fingers, bad face",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    image

    image.save(f'results/sd3_infer_result_better_prompt/sd3_infer_{i}.png')