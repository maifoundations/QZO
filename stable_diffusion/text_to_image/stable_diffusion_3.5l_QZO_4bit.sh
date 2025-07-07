export PYTHONIOENCODING=utf-8
export PYTHONPATH=/QZO/stable_diffusion:$PYTHONPATH

GPU=${GPU:-0}
TRAIN_BS=${TRAIN_BS:-8}
LR=${LR:-1e-5}
DATASET=${DATASET:-"KAKIZHOU/CUB-200"}
EPS=${EPS:-1e-3}
OUT_DIR=${OUT_DIR:-"./output"}
TIMESTEP=${TIMESTEP:-0}

TRAIN_STEP=${TRAIN_STEP:-10000}

accelerate launch --mixed_precision="fp16" --gpu_ids=$GPU /QZO/stable_diffusion/train_text_to_image.py \
  --mixed_precision="fp16"\
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large"\
  --dataset_name=$DATASET\
  --resolution=512\
  --center_crop\
  --random_flip\
  --train_batch_size=$TRAIN_BS\
  --gradient_accumulation_steps=1\
  --gradient_checkpointing\
  --max_train_steps=$TRAIN_STEP\
  --learning_rate=$LR\
  --max_grad_norm=1\
  --lr_scheduler="constant"\
  --lr_warmup_steps=0\
  --output_dir=$OUT_DIR\
  --trainer=zo\
  --zo_eps=$EPS\
  --quantized_model_finetune=True\
  --quantize_method="bitsandbytes"\
  --start_timestep=$TIMESTEP