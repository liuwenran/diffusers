export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"

accelerate launch examples/text_to_image/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir='/mnt/petrelfs/liuwenran/datasets/changshiban/train_dataset' \
  --resolution=1024 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=3000 \
  --use_8bit_adam \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="changshiban, an old man with black hat" \
  --validation_epochs 5 \
  --checkpointing_steps=1000 \
  --output_dir="work_dirs/t2i-changshiban-trainckpt-e4-fp16"