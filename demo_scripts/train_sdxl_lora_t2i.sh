export MODEL_NAME="/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
export VAE_PATH="/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15"

python examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --train_data_dir='/mnt/petrelfs/liuwenran/datasets/changshiban/train_dataset' \
  --caption_column="text" \
  --mixed_precision="fp16" \
  --resolution_h=720 \
  --resolution_w=1280 \
  --center_crop \
  --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=5 \
  --checkpointing_steps=200 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="work_dirs/t2i-changshiban/t2i-changshiban-fullsize720-e5" \
  --train_text_encoder \
  --validation_prompt="changshiban, an old man with black hat" \
  --report_to="wandb" \