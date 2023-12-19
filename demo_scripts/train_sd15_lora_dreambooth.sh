export MODEL_NAME="/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--Linaqruf--anything-v3.0/snapshots/8323d54dcf89c90c39995b04ae43166520e8992a"
export INSTANCE_DIR="/mnt/petrelfs/liuwenran/datasets/cctv/qianqiugesong/qianqiu/willow"
export OUTPUT_DIR="work_dirs/test-lora15"

accelerate launch examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --instance_prompt="a photo in chinese cartoon style, a man and a woman, willow" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo in chinese cartoon style, an old man" \
  --validation_epochs=50 \
  --seed="0" 