import json
import numpy as np
import os
os.chdir(os.path.dirname(__file__))
from diversity_metric_utils import Diversity, ExpectationAdjustedDistinctNgrams, ProcessInputFormat, SentenceBERTSim
from fast_bleu import SelfBLEU
import argparse

def main(file):
    name = file.split('/')[-1]
    samples = []
    with open(file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    per_input = ProcessInputFormat(samples)

    # Per-Input
    # 1. Diversity
    metric = Diversity(range(1, 5 + 1))
    results= []
    for sample in per_input:
        results.append(metric.measure_repetition_and_diversity(sample)[0])
    print(f"{name} Diversity: ", sum(results) / len(results))
    
    # 2. EAD
    results = []
    metric = ExpectationAdjustedDistinctNgrams(range(1, 5 + 1))
    for sample in per_input:
        results.append(metric(sample))
    print(f"{name} EAD: ", np.mean(results))
    
    # 3. FastBLEU
    selfbleu = []
    for sample in per_input:
        sentences = [t.split() for t in sample]
        bleu = SelfBLEU(sentences).get_score()
        bleu_score = 0
        for k, v in bleu.items():
            bleu_score += np.mean(v)
        selfbleu.append(np.mean(bleu_score))
    print(f"{name} BLEU: ", np.mean(selfbleu))
        
    # 4. SentBERT
    metric = SentenceBERTSim()
    sentbert_score = [metric(t) for t in per_input]
    print(f"{name} SentBERT: ", np.mean(sentbert_score))
    print("="*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the file, multiple files separated by commas.")
    args = parser.parse_args()
    evaluate_list = args.file.split(',')
    for file in evaluate_list:
        main(file)