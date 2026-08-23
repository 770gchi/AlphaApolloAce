"""Deterministic delayed-evidence vector environment for Task C."""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np


class DelayedEvidenceVectorEnv:
    def __init__(self, seed=0, env_num=1, group_n=1, is_train=False, env_config=None):
        self.batch_size = int(env_num) * int(group_n)
        self.max_steps = int(env_config.max_steps)
        self.cases: List[Dict[str, Any]] = []
        self.cursors: List[int] = []

    def reset(self, kwargs: List[Dict[str, Any]]):
        if kwargs is None or len(kwargs) > self.batch_size:
            raise ValueError("delayed_evidence reset requires one env_kwargs entry per case")
        self.cases = [deepcopy(item["case"]) for item in kwargs]
        self.cursors = [0 for _ in self.cases]
        observations = [case["initial_goal"] for case in self.cases]
        infos = [{"data_source": "delayed_evidence", "case_id": case["case_id"]} for case in self.cases]
        return observations, infos

    def step(self, actions: List[str], text_actions: List[str]):
        observations, rewards, dones, infos = [], [], [], []
        for index, case in enumerate(self.cases):
            cursor = self.cursors[index]
            raw_action = str(text_actions[index] if index < len(text_actions) else actions[index])
            script = case["scripted_observations"]
            if cursor < len(script):
                item = script[cursor]
                self.cursors[index] += 1
                observations.append(item["content"])
                rewards.append(0.0)
                dones.append(False)
                infos.append({
                    "won": False,
                    "data_source": "delayed_evidence",
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "length_tier": case["length_tier"],
                    "script_step": item["step_id"],
                    "step_tags": item["tags"],
                })
            else:
                correct = re.fullmatch(case["answer_regex"], raw_action.strip(), flags=re.IGNORECASE) is not None
                observations.append("")
                rewards.append(1.0 if correct else 0.0)
                dones.append(True)
                infos.append({
                    "won": correct,
                    "data_source": "delayed_evidence",
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "length_tier": case["length_tier"],
                    "expected_answer": case["expected_answer"],
                    "final_action": raw_action,
                })
        return observations, np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), infos

    def close(self):
        return None


def build_delayed_evidence_envs(seed=0, env_num=1, group_n=1, is_train=False, env_config=None):
    return DelayedEvidenceVectorEnv(seed, env_num, group_n, is_train, env_config)
