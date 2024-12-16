# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import deepspeed
from transformers import StoppingCriteria, StoppingCriteriaList
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from dschat.utils.utils import print_rank_0, unwrap_model_for_generation, KMP_search, SelfBLEURewardFunction

def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any(stop_string in decoded_generation for stop_string in self.eof_strings))
        return all(done)

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.ICM = self.rlhf_engine.ICM
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # In case the generated experience is not valid (too short), we use the last valid
        # generated experience. Alternatively, we can skip the step (on all workers).
        # For now, use the last valid experience which is a simpler solution
        self.last_generated_experience = None

        # Those value can be changed
        self.kl_ctl = self.args.kl_ctl
        self.clip_reward_value = 2.5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = self.args.gamma
        self.lam = self.args.lam
        self.generate_time = 0.0
        self.temperature = self.args.temperature
        self.eta = self.args.eta
        self.cdrlhf_topk = self.args.cdrlhf_topk

        self.selfbleu_reward = SelfBLEURewardFunction(self.tokenizer, self.args.sample_size)

    def _generate_sequence(self, prompts, mask, step):
        # clean cuda cache
        torch.cuda.empty_cache()
        
        prompt_length = prompts.shape[1]
        max_min_length = self.max_answer_seq_len + prompt_length

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        generation_config = dict(do_sample=True, temperature=self.temperature)
        with torch.no_grad():
            if self.args.enable_zero3_generation_gather and self.z3_enabled:
                with unwrap_model_for_generation(self.actor_model) as unwrapped_model:
                    output = unwrapped_model.generate(
                        prompts,
                        attention_mask=mask,
                        max_length=max_min_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        synced_gpus=self.z3_enabled,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **generation_config)
            else:
                output = self.actor_model.module.generate(
                    prompts,
                    attention_mask=mask,
                    max_length=max_min_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    synced_gpus=self.z3_enabled,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_config)
        
        seq = output.sequences
        
        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        
        logits = torch.stack(output.scores, dim=0).transpose(0, 1)
        probs = logits.softmax(dim=-1)[:, :ans.size(1), :] # bsz, seq_len, vocab_size
        _, top_indices = torch.topk(probs, self.cdrlhf_topk, dim=-1)
        in_top_k = ~((ans.unsqueeze(-1) == top_indices).any(dim=-1))

        if self.args.print_answers and (step % self.args.print_answers_interval
                                        == 0):
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        out_in_top_k = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.decode(prompts[i], skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.decode(ans[i], skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])
                out_in_top_k.append(in_top_k[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None, None

        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim
        out_in_top_k = torch.cat(out_in_top_k, dim=0)

        return out_seq, out_in_top_k

    def generate_experience(self, prompts, mask, step):
        self.eval()
        generate_start = time.time()
        seq, in_top_k = self._generate_sequence(prompts, mask, step)
        generate_end = time.time()
        if seq is None:
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
            in_top_k = self.last_generated_experience['in_top_k']
        else:
            self.last_generated_experience = {'prompts': prompts, 'seq': seq, 'in_top_k': in_top_k}
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
            output_ref = self.ref_model(seq, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

            # SelfBLEU metric: https://arxiv.org/abs/2402.19464
            bleu_score = self.selfbleu_reward.compute_bleu(seq[:, prompts.size(1):]).to(self.actor_model.module.device)
            max_len = 512
            padded_seq = torch.zeros(seq.size(0), max_len, dtype=seq.dtype).to(seq.device)
            padded_seq[:, :seq.size(1) - prompts.size(1)] = seq[:, prompts.size(1):]
            gathered_seqs = [torch.zeros(seq.size(0), max_len, dtype=seq.dtype).to(seq.device) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_seqs, padded_seq)
            gathered_seqs = torch.cat(gathered_seqs, dim=0).to('cpu')
            self.selfbleu_reward.add_sample(gathered_seqs)
            del gathered_seqs

        hidden_states = output_ref.hidden_states[-1]
        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        self.generate_time = generate_end - generate_start

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards': reward_score,
            'bleu_rewards': bleu_score,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'hidden_states': hidden_states,
            'intrinsic_mask': in_top_k,
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask, intrinsic_reward):
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = torch.clone(kl_divergence_estimate)
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)

        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
            rewards[j, start:ends[j]] += self.eta * intrinsic_reward[j, :ends[j] - start]

        return rewards, kl_divergence_estimate

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        hidden_states = inputs['hidden_states']
        intrinsic_mask = inputs['intrinsic_mask']
        
        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        
        resp_length = torch.tensor(seq[:, start:].size(1)).to(seq)

        old_values = values
        
        # ICM Loss and intrinsic reward
        action = self.actor_model.module.get_input_embeddings()(seq[:, start + 1:]).detach()

        next_state, next_state_hat = self.ICM(hidden_states[:, start: -1, :], hidden_states[:, start + 1:, :], action)

        icm_loss = self.ICM_loss(next_state, next_state_hat, action_mask[:, start:])
        intrinsic_reward = self.intrinsic_reward(next_state, next_state_hat, action_mask[:, start:]) * intrinsic_mask
        
        self.ICM.backward(icm_loss)

        with torch.no_grad():
            old_rewards, kl_divergence = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score,
                                               action_mask, intrinsic_reward=intrinsic_reward)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        # kl_loss, kl_divergence = self.compute_kl(log_probs[:, start:], ref_log_probs[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()
        
        self.ICM.step()

        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'icm_loss': icm_loss,
            'kl': kl_divergence.norm(p=2, dim=-1).mean(),
            'reward': reward_score.mean(),
            'bleu_reward':inputs.get("bleu_rewards", 0).mean(),
            'resp_length': resp_length,
            'intrinsic_rewards': intrinsic_reward.norm(p=2, dim=-1).mean(),
            'entropy': -log_probs.sum(-1).mean(),
            'advantages_mean': advantages.mean(),
            'advantages_std': advantages.std(),
            'advantages_norm': advantages.norm(p=2, dim=-1).mean(),
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
            'returns_norm': returns.norm(p=2, dim=-1).mean(),
            'intrinsic_number': intrinsic_mask.int().sum()
        }

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def ICM_loss(self, next_state, next_state_hat, mask):
        forward_model_loss = 0.5 * torch.sum((next_state_hat - next_state).norm(2, dim=-1) * mask) / mask.sum()
        return forward_model_loss

    def whiten(self, values, shift_mean=True):
        mean, var = torch.mean(values), torch.var(values, unbiased=False)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened

    def intrinsic_reward(self, next_state, next_state_hat, mask):
        intrinsic_reward = 0.5 * (next_state - next_state_hat).norm(2, dim=-1) * mask
        intrinsic_reward = self.whiten(intrinsic_reward)
        intrinsic_reward = intrinsic_reward.detach()
        return intrinsic_reward

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
