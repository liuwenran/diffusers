# export MODEL_NAME="black-forest-labs/FLUX.1-dev"
# export INSTANCE_DIR="dog"
# export OUTPUT_DIR="trained-flux-lora"

# accelerate launch train_dreambooth_lora_flux.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --mixed_precision="bf16" \
#   --instance_prompt="a photo of sks dog" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --guidance_scale=1 \
#   --gradient_accumulation_steps=4 \
#   --optimizer="prodigy" \
#   --learning_rate=1. \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of sks dog in a bucket" \
#   --validation_epochs=25 \
#   --seed="0" \
#   --push_to_hub

export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="work_dirs/cctv/qianqiushisong_s3/flux_shouhui_lora_e3"
export INSTANCE_DIR="/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong_s3/styles/shouhui"

# accelerate launch  --main_process_port 29100  examples/dreambooth/train_dreambooth_lora_flux.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --mixed_precision="bf16" \
#   --instance_prompt="a photo in chinese cartoon style, characters in animations" \
#   --resolution=1024 \
#   --train_batch_size=1 \
#   --guidance_scale=1 \
#   --gradient_accumulation_steps=4 \
#   --optimizer="prodigy" \
#   --learning_rate=1. \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=5000 \
#   --validation_prompt="a photo in chinese cartoon style, an old man" \
#   --validation_epochs=50 \
#   --seed="0"

accelerate launch  --main_process_port 29100  examples/dreambooth/train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo in chinese cartoon style, characters in animations" \
  --resolution=1024 \
  --train_text_encoder \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-3 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="a photo in chinese cartoon style, an old man" \
  --validation_epochs=50 \
  --seed="0"