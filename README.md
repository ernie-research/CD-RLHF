# [Curiosity-Driven Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2501.11463)

  <a href="https://arxiv.org/abs/2501.11463" target="_blank">
      <img alt="Paper" src="https://img.shields.io/badge/📜-Paper-purple" />
   </a>

This repository contains the source code for reproducing the [CD-RLHF](https://arxiv.org/pdf/2501.11463). We implement CD-RLHF based on DeepSpeed-Chat.

## Get Started

### Prerequisites

We provide the environments used in our experiments in `requirements.txt`, which can be easily installed with
```bash
cd CD-RLHF
pip install -r requirements.txt
```
We use transformers==4.46.3 when training Llama-3.2 models.

### Installation

Installing our implemented deepspeed-chat:
```bash
cd CD-RLHF/applications/DeepSpeed-Chat
pip install -e .
```

## Dataset

We conduct experiments on two datasets: [OpenAI TL;DR](https://huggingface.co/datasets/openai/summarize_from_feedback), and [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized). 

We split each dataset into three parts to carry out Supervised Fine-Tuning (SFT), reward modeling, and Reinforcement Learning with Human Feedback (RLHF) fine-tuning. The data is partitioned as follows: 20% for SFT, 40% for reward modeling, and 40% for RLHF fine-tuning. This partitioning is automated using Deepspeed-Chat codes with a random seed set to 1234.

## Training 

### SFT Training

```shell
cd applications/DeepSpeed-Chat/training/step1_supervised_finetuning && bash training_scripts/tldr/run_gemma_2b.sh
```
This command train the SFT model with Gemma-2B model on TL;DR datasets.

### Reward Modeling

```shell
cd applications/DeepSpeed-Chat/training/step2_reward_model_finetuning && bash training_scripts/tldr/run_gemma_wb.sh
```
This command train the reward model with Gemma-2B model on TL;DR datasets.

### PPO Training

```shell
cd applications/DeepSpeed-Chat/training/step3_rlhf_finetuning && bash training_scripts/tldr/run_gemma_2b.sh
```
This command train the RLHF model with Gemma-2B model on TL;DR datasets using vanilla RLHF.

```shell
cd applications/DeepSpeed-Chat/training/step3_rlhf_finetuning && bash training_scripts/tldr/run_gemma_2b_cdrlhf.sh
```
This command train the RLHF model with Gemma-2B model on TL;DR datasets using CD-RLHF.

In the scripts, the $\eta=0.0$ corresponds to vanilla RLHF, and a non-zero value corresponds to CD-RLHF. By default, we keep the learning rate of intrinsic curiosity module to 1e-5, and use top-1 to enable curiosities. These hyper-parameters can be changed using `--icm_learning_rate` and `--cdrlhf_topk`.

## Inference

We provide a Python file for inference located at `CD-RLHF/evaluation/reward_model_score.py`. In this file, we randomly select 2000 instances from the validation sets (using seed 1234) for inference. The generated responses are then scored with the reward model.

You can run the inference script with the following command:
```bash
cd MA-RLHF
python evaluation/reward_model_score.py \
    --dataset-path openai/summarize_from_feedback \
    --model-path ${actor_model_path} \
    --model-name gemma-2b-tldr-rlhf \ 
    --reward-model ${reward_model_path} \
    --gpus 0,1,2,3,4,5,6,7 \
    --batch-size 16
```
The `--dataset` argument can be chosen from `[openai/summarize_from_feedback, HuggingFaceH4/ultrafeedback_binarized]`, corresponding to openai/summarize_from_feedback and HuggingFaceH4/ultrafeedback_binarized, respectively. The inference results will be saved in `./results/rewards/summarize_from_feedback/gemma-2b-tldr-rlhf.jsonl`.

Besides, we provide `CD-RLHF/evaluation/generate_samples.py` for diversity metric evaluation, which randomly selects 500 samples and generates 10 responses per prompt.
```bash
cd MA-RLHF
python evaluation/generate_samples.py \
    --dataset-path openai/summarize_from_feedback \
    --model-path ${actor_model_path} \
    --model-name gemma-2b-tldr-rlhf \ 
    --gpus 0,1,2,3,4,5,6,7 \
```
The args are same as the one used in `reward_model_score.py`. The results will be saved in `./results/generated/summarize_from_feedback/gemma-2b-tldr-rlhf.jsonl`.

## Evaluation

### RM Scores

The RM scores have already been computed during the inference stage.

### Diversity

We implement the four diversity metrics used in our paper: Diversity, EAD, SelfBLEU, SentBERT. These metrics can be calculated using `./evaluation/diversity_eval.py`:
```bash
python evaluation/diversity_eval.py \
    --file results/generated/summarize_from_feedback/gemma-2b-tldr-rlhf.jsonl,results/generated/summarize_from_feedback/gemma-2b-tldr-cdrlhf.jsonl
```
The `--file` can be multiple files seperated by comma.

### GPT-4 Evaluation

For GPT-4 evaluations, we randomly select 50 instances from the inference results. This can be done with the following command:
```bash
python evaluation/sample_from_dataset.py \
    --data-path ./results/generated/${dataset}/${actor_model_name}.jsonl \
    --save-path ./results/generated/${dataset}/${actor_model_name}-sampled.jsonl \
    --dataset summarize
```
Specifically, we select 50 instances for the TL;DR dataset based on the provided SubReddit information with `--dataset summarize`, and 50 instances for the UltraFeedback datasets by random with `--dataset ultrafeedback`.

The GPT-4 evaluation results can be obtained using:
```bash
python evaluation/gpt4-eval.py \
    --model_name_a ./results/generated/${dataset}/${actor_model_name_a}-sampled.jsonl \
    --model_name_b ./results/generated/${dataset}/${actor_model_name_b}-sampled.jsonl \
    --output ${PROJ_PATH}/results/${dataset}/${actor_model_name_a}-v.s.-${actor_model_name_b}.jsonl \
    --sk ${OPENAI_SK}
```

## Citation
```
@misc{sun2025curiositydrivenreinforcementlearninghuman,
      title={Curiosity-Driven Reinforcement Learning from Human Feedback}, 
      author={Haoran Sun and Yekun Chai and Shuohuan Wang and Yu Sun and Hua Wu and Haifeng Wang},
      year={2025},
      eprint={2501.11463},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.11463}, 
}
```
