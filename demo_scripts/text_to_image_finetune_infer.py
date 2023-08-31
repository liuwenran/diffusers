from accelerate import Accelerator
from diffusers import DiffusionPipeline

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "resources/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id)

accelerator = Accelerator()

# Use text_encoder if `--train_text_encoder` was used for the initial training
unet = accelerator.prepare(pipeline.unet)

# Restore state from a checkpoint path. You have to use the absolute path here.
accelerator.load_state("sd-pokemon-model/checkpoint-7000")

# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    unet=accelerator.unwrap_model(unet),
)

# Perform inference, or save, or push to the hub
# pipeline.save_pretrained("sd-pokemon-finetuned")
pipeline.to('cuda')
image = pipeline(prompt="A rose with big eyes and smiling face").images[0]
image.save("pokemon-rose.png")