OUTPUT=./models/gemma-7b-tldr-rm
mkdir -p $OUTPUT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --model_name_or_path ./models/gemma-7b-tldr-sft \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 1024 \
   --learning_rate 1e-6 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.05 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 2 \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
   --eval_interval 100 \
   &> $OUTPUT/training.log