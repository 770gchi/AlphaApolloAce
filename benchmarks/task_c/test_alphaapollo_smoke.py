from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verl import DataProto
from benchmarks.task_c.build import build
from alphaapollo.core.environments import make_envs
from alphaapollo.core.generation.multi_turn_rollout import TrajectoryCollector


class ScriptedWorker:
    world_size = 1

    def __init__(self, tokenizer, final_answer: str, script_steps: int):
        self.tokenizer = tokenizer; self.final_answer = final_answer; self.script_steps = script_steps; self.calls = 0

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.calls += 1
        text = f"<ack>step-{self.calls}</ack>" if self.calls <= self.script_steps else f"<answer>{self.final_answer}</answer>"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        response = torch.full((len(prompts), 32), self.tokenizer.pad_token_id, dtype=torch.long)
        response[:, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        result = {key: value for key, value in prompts.batch.items()}
        result["responses"] = response
        output = DataProto.from_single_dict(result)
        output.meta_info = dict(prompts.meta_info)
        return output


def test_alphaapollo_delayed_evidence_mock_rollout():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    with tempfile.TemporaryDirectory() as tmp:
        case = build(tmp, "Qwen/Qwen2.5-3B-Instruct", case_id="exact_number_01_short")[0]
    cfg = OmegaConf.create({
        "data": {"max_prompt_length": 2048, "truncation": "right", "return_raw_chat": False, "train_batch_size": 1, "val_batch_size": 1},
        "env": {"env_name": "delayed_evidence", "seed": 0, "max_steps": 17, "history_length": 16, "resources_per_worker": {"num_cpus": 0.1}, "rollout": {"n": 1}},
        "algorithm": {"filter_groups": {"enable": False}},
        "context_policy": {"mode": "truncate", "max_input_tokens": 2048, "reserve_output_tokens": 512, "model_context_window": 32768, "keep_last_raw": True, "abstraction_cache": True, "orchestration": "heuristic", "truncate_window_steps": 4, "abstraction_max_tokens": 256, "orchestration_max_tokens": 512, "allowed_types": ["raw", "abstract", "drop"], "audit_dir": None, "store_dir": None},
    })
    env_kwargs = np.array([{"case": case}], dtype=object)
    gen_batch = DataProto.from_single_dict({
        "input_ids": torch.zeros((1, 1), dtype=torch.long), "attention_mask": torch.ones((1, 1), dtype=torch.long), "position_ids": torch.zeros((1, 1), dtype=torch.long),
        "raw_prompt": np.array([[{"role": "user", "content": case["initial_goal"]}]], dtype=object), "data_source": np.array(["delayed_evidence"], dtype=object), "env_kwargs": env_kwargs,
    })
    gen_batch.meta_info = {"eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "do_sample": False, "validate": False}
    envs, _ = make_envs(cfg); worker = ScriptedWorker(tokenizer, case["expected_answer"], len(case["scripted_observations"]))
    collector = TrajectoryCollector(cfg, tokenizer)
    result = collector.multi_turn_loop(gen_batch, worker, envs, is_train=False)
    assert worker.calls == len(case["scripted_observations"]) + 1
    assert any(item.non_tensor_batch.get("context_audit") for item in result)
    telemetry = [item.non_tensor_batch.get("truncation_telemetry") for item in result]
    assert all(event is not None for event in telemetry)
    assert all({"pre_truncation_tokens", "post_truncation_tokens", "truncation_direction", "truncated_tokens", "was_truncated"} <= set(event) for event in telemetry)
    assert float(result[0].non_tensor_batch["episode_rewards"]) == 1.0
    envs.close()


if __name__ == "__main__":
    test_alphaapollo_delayed_evidence_mock_rollout(); print("test_alphaapollo_delayed_evidence_mock_rollout: OK")
