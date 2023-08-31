export MODEL_NAME="resources/stable-diffusion-v1-5"
# export MODEL_NAME="resources/instruct-pix2pix"
export INSTANCE_DIR="/nvme/liuwenran/datasets/cultural_relic/antiques_crawl/images"
# export INSTANCE_DIR="/nvme/liuwenran/datasets/walrus_picked"
export OUTPUT_DIR="ckpts/sd-dreambooth-songdaitaoci"

accelerate launch examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of songdaitaoci" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800

  # --class_prompt="a photo of face and body" \
