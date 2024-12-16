#!/bin/bash
OUTPUT=./models/gemma-2b-ultrafeedback-sft
mkdir -p $OUTPUT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed main.py \
   --data_path HuggingFaceH4/ultrafeedback_binarized \
   --data_split 2,4,4 \
   --model_name_or_path google/gemma-2b \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 5e-5 \
   --weight_decay 0.01 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 4 \
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
