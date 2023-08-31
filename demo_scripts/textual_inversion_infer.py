from diffusers import StableDiffusionPipeline

model_id = "ckpts/sd-textual-inversion-songdaitaoci"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)

prompt = "A photo of <songdaitaoci> on the table"

for i in range(6):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save("results/textualinversion-songdaitaoci/songdaitaoci_ontable_" + str(i) + ".png")