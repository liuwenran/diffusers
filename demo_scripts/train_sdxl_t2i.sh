# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export MODEL_NAME="/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
# export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export VAE_NAME="/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15"

accelerate launch --num_processes 1 examples/text_to_image/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir='/mnt/petrelfs/share_data/liuwenran/changshiban/train_dataset' \
  --resolution=1024 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=3000 \
  --use_8bit_adam \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="changshiban, an old man with black hat" \
  --validation_epochs 1 \
  --checkpointing_steps=1000 \
  --output_dir="work_dirs/test"