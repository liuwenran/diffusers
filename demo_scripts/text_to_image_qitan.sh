export MODEL_NAME="resources/stable-diffusion-v1-5"
export TRAIN_DIR="/nvme/liuwenran/datasets/video_processed_01"
export OUTPUT_DIR="sd-qitan-model"

accelerate launch examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=5000 \
  --output_dir=${OUTPUT_DIR}