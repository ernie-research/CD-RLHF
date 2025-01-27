from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from dschat.utils.data.raw_datasets import OpenAISummarizeDataset, UltrafeedbackDataset
from tqdm import tqdm
import os
import json
import sys
import torch
import torch.multiprocessing as mp

def generate_samples(model, tokenizer, prompts):
    prompt = tokenizer.batch_encode_plus(prompts, return_tensors="pt", truncation=True, max_length=2048, padding=True).to(model.device)
    input_ids = prompt.input_ids.repeat(10, 1) # sample 10 responses for each prompt

    generated = model.generate(input_ids, do_sample=True, temperature=1.0, max_new_tokens=512)
        
    outputs = [tokenizer.decode(o[prompt["input_ids"].size(1):], skip_special_tokens=True) for o in generated]
    return outputs

def inference_worker(rank, gpu, args, prompts, return_dict):
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/actor", device_map=f'cuda:{gpu}')
        model = AutoModelForCausalLM.from_pretrained(f"{args.model_path}/actor", device_map=f"cuda:{gpu}", torch_dtype=torch.float16)
        inferences = []
        model.eval()
        for i in tqdm(range(0, len(prompts))):
            responses = []
            response = generate_samples(model, tokenizer, prompts[i: i + 1])
            responses.extend(response)
            inferences.append({"prompt": prompts[i], "sample_id": i + gpu * (1000 // len(args.gpus.split(','))), "response_list": responses})
        return_dict[rank] = inferences

def get_prompts(dataset, num_samples, start_id=0):
    if "summarize" in dataset:
        raw_dataset = OpenAISummarizeDataset("", 1234, 0, dataset)
    elif "ultrafeedback" in dataset:
        raw_dataset = UltrafeedbackDataset("", 1234, 0, dataset)
        
    validation = raw_dataset.get_eval_data()
    validation = validation.shuffle(seed=1234)
    prompts = []
    for sample in validation:
        prompts.append(raw_dataset.get_prompt(sample))
    num_samples = 500
    prompts = prompts[start_id: start_id + num_samples]
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the actor model.")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model for saving results.")
    parser.add_argument("--dataset-path", type=str, default="openai/summarize_from_feedback", help="Path to the dataset.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="GPUs to use for evaluation, separated by commas.")
    args = parser.parse_args()

    torch.manual_seed(1234)
    if args.model_name is None:
        args.model_name = args.model_path.split('/')[-2]
    prompts = get_prompts(args.dataset_path, 500)

    gpus = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpus)
    data_chunks = [prompts[i::num_gpus] for i in range(num_gpus)]

    mp.set_start_method('spawn', force=True)

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank, gpu in enumerate(gpus):
        data_chunk = data_chunks[rank]
        p = mp.Process(target=inference_worker, args=(rank, gpu, args, data_chunk, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    inferences = []
    for key in return_dict.keys():
        inferences.extend(return_dict[key])

    dataset_name = args.dataset_path.split('/')[-1]
    os.makedirs(f'./results/generated/{dataset_name}', exist_ok=True)
    with open(f'./results/generated/{dataset_name}/{args.model_name}.jsonl', 'w') as f:
        for inference in inferences:
            f.write(json.dumps(inference) + '\n')
