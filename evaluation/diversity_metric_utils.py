from typing import List
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

class Diversity():
    def __init__(self, ngrams):
        self.ngrams = ngrams
        
    def eval_text(self, text, ngram):
        token_list = text.strip().split()
        start_idx, end_idx = 0, ngram
        total_num = 0
        ngram_set = set()
        while end_idx < len(token_list):
            one_ngram_list = token_list[start_idx:end_idx]
            assert len(one_ngram_list) == ngram
            one_ngram = ' '.join(one_ngram_list)
            total_num += 1
            ngram_set.add(one_ngram)
            start_idx += 1
            end_idx += 1
        return len(ngram_set), total_num

    def eval_instance(self, text_list, ngram_list):
        res_dict = {}
        for n in ngram_list:
            text = ' '.join(text_list)
            text = text.strip('\n').strip()
            n_unique, n_total = self.eval_text(text, n)
            res_dict[n] = {'unique':n_unique, 'total':n_total}
        unique_token_set = set(text.strip().split())
        return res_dict, unique_token_set

    def measure_repetition_and_diversity(self, text):
        '''
            text_list: the list of text
        '''
        
        pred_res_dict = {}
        for n in self.ngrams:
            pred_res_dict[n] = {}
            pred_res_dict[n]['unique'] = 0
            pred_res_dict[n]['total'] = 0
        pred_unique_token_set = set()

        one_pred_res_dict, one_pred_uni_token_set = self.eval_instance(text, self.ngrams)

        # unique token set
        pred_unique_token_set = pred_unique_token_set.union(one_pred_uni_token_set)
        # ngram statistic
        for n in self.ngrams:
            pred_res_dict[n]['unique'] += one_pred_res_dict[n]['unique']
            pred_res_dict[n]['total'] += one_pred_res_dict[n]['total']

        # prediction result
        pred_seq = {}
        for n in self.ngrams:
            if pred_res_dict[n]['total'] == 0:
                pred_seq_n = 0
                pred_seq_n = round(pred_seq_n * 100, 2)
            else:
                pred_seq_n = 1 - (pred_res_dict[n]['unique']/pred_res_dict[n]['total'])
                pred_seq_n = round(pred_seq_n * 100, 2)
            pred_seq[n] = pred_seq_n
        pred_div = 1
        for n in self.ngrams:
            pred_div *= (1 - pred_seq[n]/100)

        return pred_div, pred_res_dict

def lines_to_ngrams(lines, n=3):
    ngrams = []
    for s in lines:
        words = [e for e in s.replace(".", "").replace("\n", "").split(" ") if e != ""]
        ngrams.append([tuple(words[i : i + n]) for i in range(len(words) - n + 1)])
    return ngrams

class ExpectationAdjustedDistinctNgrams():
    # Taken from https://arxiv.org/abs/2202.13587
    name = "ead_averaged_distinct_ngrams"

    def __init__(self, ngrams):
        self.vocab_size = 256000
        self.ngrams = ngrams

    def ead_normalized_unique_ngrams(self, ngram_lists):
        """
        Calc expectation-adjusted portion of unique n-grams out of all n-grams.
        :param ngram_lists: list of lists of ngrams
        :return: value in (0,1]
        """
        ngrams = [item for sublist in ngram_lists for item in sublist]
        N = len(set(ngrams))
        C = len(ngrams)
        V = self.vocab_size

        try:
            ead = N / (V * (1 - ((V - 1) / V) ** C))
        except ZeroDivisionError:
            ead = 0.0
        return ead

    def __call__(self, response_set):
        results = []
        for n in self.ngrams:
            results.append(self.ead_normalized_unique_ngrams(lines_to_ngrams(response_set, n=n)))
        return sum(results) / len(self.ngrams)

class SentenceBERTSim():
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def __call__(self, text_list: List[str]):
        encoded_text = self.model.encode(text_list, convert_to_tensor=True, convert_to_numpy=False)
        encoded_text = F.normalize(encoded_text, dim=1, p=2)
        cosine_sim = torch.mm(encoded_text, encoded_text.T).fill_diagonal_(0)
        element_sim = cosine_sim.sum(dim=1) / (cosine_sim.size(-1) - 1)
        return element_sim.cpu().numpy()

def ProcessInputFormat(data: List[dict]) -> List[List[str]]:
    return [d['response_list'] for d in sorted(data, key=lambda x: int(x['sample_id']))]