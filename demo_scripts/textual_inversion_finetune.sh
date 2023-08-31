export MODEL_NAME="resources/stable-diffusion-v1-5"
export DATA_DIR="/nvme/liuwenran/datasets/cultural_relic/antiques_crawl/images"

accelerate launch examples/textual_inversion/textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<songdaitaoci>" --initializer_token="ceramics" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="ckpts/sd-textual-inversion-songdaitaoci"