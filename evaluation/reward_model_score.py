import argparse
import os
import json
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dschat.utils.data.raw_datasets import OpenAISummarizeDataset, UltrafeedbackDataset
from dschat.utils.model.model_utils import create_critic_model

def generate_response(model, tokenizer, temperature, inputs):
    generate_kwargs = {
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
    }
    generated = model.generate(**inputs, max_new_tokens=512, **generate_kwargs)
        
    outputs = [tokenizer.decode(o[inputs["input_ids"].size(1):], skip_special_tokens=True) for o in generated]
    return outputs

def get_reward_score(reward_model, tokenizer, prompts, responses, prompt_length):
    tokenizer.padding_side = 'right'
    inputs = tokenizer.batch_encode_plus([p + r + tokenizer.eos_token for p, r in zip(prompts, responses)], return_tensors="pt", truncation=True, max_length=1024, padding=True).to(reward_model.rwtransformer.device)
            
    reward_score = reward_model.forward_value(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], prompt_length=prompt_length)['chosen_end_scores']
    tokenizer.padding_side = 'left'
    return reward_score

def process_data(rank, prompts, chosens, rejecteds, args, return_dict):
    torch.manual_seed(1234)
    
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/actor", device_map='cuda')
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(f"{args.model_path}/actor", device_map=f"cuda:{rank}", torch_dtype=torch.float16)
    generated_results = []
    
    inferences = []
    model.eval()
    
    batch_size = args.batch_size
    if not args.ref_scores:
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Rank {rank}"):
                inputs = tokenizer.batch_encode_plus(prompts[i: i + batch_size], return_tensors="pt", truncation=True, max_length=1024, padding=True).to(model.device)
                response = generate_response(model, tokenizer, 0.8, inputs)
                generated_results.append(response)
    del model
    torch.cuda.empty_cache()
    
    reward_model = create_critic_model(args.reward_model, tokenizer, None, rlhf_training=True).half().cuda(rank)
    reward_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Rank {rank}"):
            if args.ref_scores:
                chosen_reward_score = get_reward_score(reward_model, tokenizer, prompts[i: i + batch_size], chosens[i: i + batch_size], 2)
                rejected_reward_score = get_reward_score(reward_model, tokenizer, prompts[i: i + batch_size], rejecteds[i: i + batch_size], 2)
                
                for j in range(len(chosen_reward_score)):
                    inferences.append({"prompt": prompts[i + j], "response": chosens[i + j], "reward": chosen_reward_score[j].item(), "rejected_response": rejecteds[i + j], "rejected_reward": rejected_reward_score[j].item()})
            else:
                response = generated_results[i // batch_size]
                reward_score = get_reward_score(reward_model, tokenizer, prompts[i: i + batch_size], response, 2)
            
                for j in range(len(response)):
                    inferences.append({"prompt": prompts[i + j], "response": response[j], "reward": reward_score[j].item()})
    
    return_dict[rank] = inferences

def main():
    mp.set_start_method('spawn', force=True)  # Use 'spawn' for multiprocessing with CUDA

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the actor model.")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the actor model for saving results.")
    parser.add_argument("--reward-model", type=str, required=True, help="Path to the reward model.")
    parser.add_argument("--dataset-path", type=str, default="openai/summarize_from_feedback", help="Path to the dataset.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="GPUs to use for evaluation, separated by commas.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--ref-scores", action='store_true', help="Scoring references provided by dataset.")
    args = parser.parse_args()

    gpus = [int(gpu) for gpu in args.gpus.split(",")]

    if "summarize" in args.dataset_path:
        raw_dataset = OpenAISummarizeDataset("", 1234, 0, args.dataset_path)
    elif "ultrafeedback" in args.dataset_path:
        raw_dataset = UltrafeedbackDataset("", 1234, 0, args.dataset_path)

    validation = raw_dataset.get_eval_data().shuffle(seed=1234)
    prompts, chosens, rejecteds = [], [], []
    for sample in validation:
        prompts.append(raw_dataset.get_prompt(sample))
        chosens.append(raw_dataset.get_chosen(sample))
        rejecteds.append(raw_dataset.get_rejected(sample))

    num_samples = 2000
    prompts = prompts[:num_samples]
    chosens = chosens[:num_samples]
    rejecteds = rejecteds[:num_samples]

    # Split data into chunks for each process/GPU
    num_gpus = len(gpus)
    chunk_size = num_samples // num_gpus
    chunks = [(prompts[i:i + chunk_size], chosens[i:i + chunk_size], rejecteds[i:i + chunk_size]) for i in range(0, num_samples, chunk_size)]

    # Multiprocessing
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in gpus:
        p = mp.Process(target=process_data, args=(rank, *chunks[rank], args, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Combine results
    all_inferences = []
    for rank in gpus:
        all_inferences.extend(return_dict[rank])

    # Save to file
    dataset_name = args.dataset_path.split('/')[-1]
    os.makedirs(f"./results/rewards/{dataset_name}", exist_ok=True)
    with open(f'./results/rewards/{dataset_name}/{args.model_name}.jsonl', 'w') as f:
        for inference in all_inferences:
            f.write(json.dumps(inference) + '\n')

if __name__ == "__main__":
    main()
