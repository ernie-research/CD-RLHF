#!/bin/bash
OUTPUT=./models/gemma-7b-tldr-sft
mkdir -p $OUTPUT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --model_name_or_path google/gemma-7b \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 2048 \
   --learning_rate 2e-5 \
   --weight_decay 0.01 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.1 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 2 \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
   --print_loss \
   &> $OUTPUT/training.log
