export MODEL_NAME="resources/instruct-pix2pix"
export INSTANCE_DIR="/nvme/liuwenran/datasets/expression"
export OUTPUT_DIR="ckpts/sd-instructp2p-expression-five"

accelerate launch examples/instructp2p/train_instructp2p.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="sketch the expression" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800