#!/bin/bash
OUTPUT=./models/llama-3.2-1b-ultrafeedback-sft
mkdir -p $OUTPUT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed main.py \
   --data_path HuggingFaceH4/ultrafeedback_binarized \
   --data_split 2,4,4 \
   --model_name_or_path meta-llama/Llama-3.2-1B \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
   --learning_rate 1e-4 \
   --weight_decay 0.01 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 8 \
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
