export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="/mnt/petrelfs/liuwenran/datasets/cctv/donghuatest/胡桃桃场景"
export OUTPUT_DIR="work_dirs/cctv/lora-trained-xl-hutaotao-e4"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of scene in cartoon style, ink painting" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --train_text_encoder \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=310 \
  --checkpointing_steps=100 \
  --validation_prompt="a photo of a building in cartoon style, ink painting" \
  --validation_epochs=30 \
  --seed="0"