#!/bin/bash
OUTPUT=./models/gemma-7b-ultrafeedback-rlhf
mkdir -p $OUTPUT

echo $(basename $OUTPUT)
branch_info=$(git branch | grep '*')
commit_info=$(git rev-parse --short HEAD)
echo "branch: $branch_info commit id: $commit_info" > $OUTPUT/training.log

Actor_Lr=1e-6
Critic_Lr=1e-6

deepspeed --num_gpus 8 main.py \
   --data_path HuggingFaceH4/ultrafeedback_binarized \
   --data_split 2,4,4 \
   --actor_model_name_or_path ./models/gemma-7b-ultrafeedback-sft \
   --critic_model_name_or_path ./models/gemma-7b-ultrafeedback-rm \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type linear \
   --gradient_accumulation_steps 32 \
   --end_of_conversation_token "<eos>" \
   --actor_dropout 0.0 \
   --warmup_ratio 0.1 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --icm_zero_stage 3 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload \
   --offload_reference_model \
   --offload_icm_model \
   --enable_zero3_generation_gather \
   --output_dir $OUTPUT \
   --icm_learning_rate 1e-5 \
   --eta 0.0 \
   --cdrlhf_topk 1 \
   --sample_size 1000 \
   --kl_ctl 0.05 \
   --actor_lora_dim 16 \
   --actor_lora_module_name q_proj,k_proj,v_proj,o_proj \
   --critic_lora_dim 16 \
   --critic_lora_module_name q_proj,k_proj,v_proj,o_proj \
   --only_optimize_lora \
   --actor_lora_learning_rate 5e-4 \
   --critic_lora_learning_rate 5e-4 \
   --print_answers \
   --print_answers_interval 50 \
   --save_steps 1000 \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
    &>> $OUTPUT/training.log
wait
bash /root/paddlejob/workspace/env_run/g/gpu_b.sh