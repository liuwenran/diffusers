export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

python examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --train_data_dir='/mnt/petrelfs/liuwenran/datasets/changshiban/train_dataset' \
  --caption_column="text" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=4 \
  --checkpointing_steps=300 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="work_dirs/t2i-changshiban-e4-fp16" \
  --logging_dir="changshiban-e5-fp16" \
  --train_text_encoder \
  --validation_prompt="changshiban, an old man with black hat" \
  --report_to="wandb" \