export MODEL_NAME="resources/stable-diffusion-v1-5"
export INSTANCE_DIR="/nvme/liuwenran/datasets/gaoqiqiang"
export OUTPUT_DIR="ckpts/sd-dreambooth-gaoqiqiang"

accelerate launch examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of gaoqiqiang head" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400