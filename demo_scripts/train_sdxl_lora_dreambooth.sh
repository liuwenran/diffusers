export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export MODEL_NAME='/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--playgroundai--playground-v2.5-1024px-aesthetic/snapshots/1e032f13f2fe6db2dc49947dbdbd196e753de573/'
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
# export INSTANCE_DIR="/mnt/petrelfs/liuwenran/datasets/cctv/qianqiushisong3/new_styles/dawangriji_scene"
# export INSTANCE_DIR="/mnt/petrelfs/liuwenran/datasets/cctv/20240415_scene_train"
export INSTANCE_DIR="/mnt/petrelfs/liuwenran/datasets/cctv/诗词动画角色汇总/character_zhengmian_extract_1080"
export OUTPUT_DIR="work_dirs/cctv/platform/lora-trained-xl-playground_human600"

accelerate launch --num_processes 1 --main_process_port 29400 examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo in chinese cartoon style, characters in animation" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --width=1080 \
  --height=1920 \
  --train_batch_size=1 \
  --train_text_encoder \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=600 \
  --checkpointing_steps=400 \
  --validation_epochs=30 \
  --seed="0"
  # --validation_prompt="scene in chinese cartoon style, village" \
