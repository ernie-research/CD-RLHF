import json
import os
import time
import re
import random
import argparse
import time
from openai import OpenAI

def get_gpt_response(query, sk):
    client = OpenAI(api_key=sk)

    SUCCESS = False
    result = "None"
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=5
            )
            result = response
            response = result.choices[0].message.content
            SUCCESS = True
            break
        except Exception as e:
            print("error:", e)
            time.sleep(2)
            continue

    if SUCCESS:
        return response
    else:
        return "None"

rnkGDS_Prompt = """Given two sets of responses, your task is to identify which set is more diverse compared to the other. The diversity evaluation should assess the variation among the proposed responses. The more similar the responses within a set, the lower the diversity. And also, you need to identify which set of responses is more related to the prompt:
Prompt: {}
Set 0: {}
Set 1: {}
Your identification (starting with reason, and end with your choice, using ``MY CHOICE: 0/1/2'' to mark the choice, where 0 for Set 0, 1 for Set 1, and 2 for equally diverse):"""

def rnkGPT_eval(comparison, reference, change_case):
    if change_case:
        input_text = rnkGDS_Prompt.format(reference['prompt'], comparison['response_list'], reference['response_list'])
        results, _ = get_gpt_response(input_text)
    else:
        input_text = rnkGDS_Prompt.format(reference['prompt'], reference['response_list'], comparison['response_list'])
        results, _ = get_gpt_response(input_text)
    print(results)
    # extract the choice from results: MY CHOICE {0,1}
    choice = re.findall(r"MY CHOICE: (\d)", results)
    choice = int(choice[0])
    if change_case:
        choice = 1 - choice
    return choice, results, reference['prompt'], reference['response_list'], comparison['response_list']

def main(args):
    random.seed(42)

    reference_file = args.model_name_a
    compare_file = args.model_name_b

    references = []
    with open(reference_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            references.append(data)

    comparisons = []
    with open(compare_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            comparisons.append(data)

    idx = [i for i in range(len(references))]

    output_dir = '/'.join(args.output.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    rated_lines = 0
    for i in idx[rated_lines:50]:
        assert references[i]['prompt'] == comparisons[i]['prompt']
        print(f"Sample {i + 1}")
        change_case = random.choice([True, False])

        choice, results, prompt, ref_resps, comp_resps = rnkGPT_eval(comparisons[i], references[i], change_case)
        with open(args.output, 'a') as f:
            json.dump({'prompt': prompt, 'model_a_resps': ref_resps, 'model_b_resps': comp_resps, 'sample': i, 'choice': choice, 'reason': results, 'change_pos': change_case}, f)
            f.write('\n')
    time.sleep(3)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_a", type=str, required=True)
    parser.add_argument("--model_name_b", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sk", type=str, required=True)
    args = parser.parse_args()
    main(args)