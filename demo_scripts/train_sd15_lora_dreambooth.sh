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

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --seed="0" 