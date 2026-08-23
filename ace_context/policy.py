"""Unified context policy: full | truncate | summarize | elastic.

This is the single entry point Task B integrates into AlphaApollo:
before each generation call, call `record_step(...)` for the new observation,
then `build(...)` to get the exact message view the main LLM should receive.

- full:      pass history through untouched (native AlphaApollo behavior)
- truncate:  fixed sliding window over raw history
- summarize: compress all old history into a single summary
- elastic:   full ACE (lossless store + orchestration + packing)

`build` also returns an AuditEntry with every field Task B requires.
"""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Dict, List, Optional, Tuple

from .abstraction import AbstractionService
from .llm import LLMClient
from .models import Abstraction, AuditEntry, Decision, ElasticType, StepRecord
from .orchestrator import ElasticOrchestrator
from .packing import BudgetExceededError, BudgetPacker
from .store import LosslessStore

VALID_MODES = ("full", "truncate", "summarize", "elastic")


class ContextPolicy:
    def __init__(
        self,
        mode: str = "full",
        max_input_tokens: int = 4096,
        reserve_output_tokens: int = 512,
        keep_last_raw: bool = True,
        llm: Optional[LLMClient] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        window_size: int = 8,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {VALID_MODES}")
        self.mode = mode
        self.max_input_tokens = max_input_tokens
        self.reserve_output_tokens = reserve_output_tokens
        self.keep_last_raw = keep_last_raw
        self.window_size = window_size
        self.llm = llm

        self.store = LosslessStore()
        self.abstraction = AbstractionService(llm=llm)
        self.orchestrator = ElasticOrchestrator(llm=llm, keep_last_raw=keep_last_raw)
        self.packer = BudgetPacker(
            count_tokens=token_counter or _default_counter,
            reserve_output_tokens=reserve_output_tokens,
        )
        self._summary: Dict[str, str] = {}  # 各 trajectory 的摘要绝不能相互污染。
        self._active_trajectory: Optional[str] = None

    # ------------------------------------------------------------------
    # record side
    # ------------------------------------------------------------------
    def record_step(
        self,
        trajectory_id: str,
        step_id: int,
        role: str,
        raw_messages: List[str],
        token_count: int = 0,
    ) -> None:
        rec = StepRecord(
            trajectory_id=trajectory_id,
            step_id=step_id,
            role=role,
            raw_messages=raw_messages,
            token_count=token_count,
        )
        self.store.append(rec)
        self._active_trajectory = trajectory_id
        if self.mode == "elastic":
            if self.store.get_abstraction(step_id, trajectory_id) is None:
                abs_ = self.abstraction.abstract(step_id, role, raw_messages)
                self.store.put_abstraction(step_id, trajectory_id, abs_)
        if self.mode == "summarize":
            self._roll_summary(trajectory_id, raw_messages)

    # ------------------------------------------------------------------
    # build side
    # ------------------------------------------------------------------
    def build(
        self,
        task: str,
        system_prompt: str = "",
        tool_schema: str = "",
        trajectory_id: Optional[str] = None,
        decision_step: Optional[int] = None,
    ) -> Tuple[List[Dict[str, str]], AuditEntry]:
        start = time.time()
        trajectory_id = trajectory_id or self._active_trajectory
        if trajectory_id is None:
            raise ValueError("build requires a trajectory_id when the store is empty")
        steps = [self.store.get(i, trajectory_id) for i in self.store.steps(trajectory_id)]
        steps = [s for s in steps if s is not None]
        latest = max((s.step_id for s in steps), default=-1)
        if decision_step is None:
            decision_step = latest

        if self.mode == "full":
            messages, decisions = self._build_full(steps, system_prompt, tool_schema, task)
        elif self.mode == "truncate":
            messages, decisions = self._build_truncate(steps, system_prompt, tool_schema, task)
        elif self.mode == "summarize":
            messages, decisions = self._build_summarize(steps, system_prompt, tool_schema, task)
        else:
            messages, decisions = self._build_elastic(task, steps, system_prompt, tool_schema, trajectory_id)

        # full 保持原始消息序列；若它本身溢出，显式失败而非悄悄截断。
        if self.mode != "elastic" and _count_messages(messages, self.packer.count_tokens) > self.max_input_tokens - self.reserve_output_tokens:
            raise BudgetExceededError(f"{self.mode} context exceeds configured budget")

        actual = _count_messages(messages, self.packer.count_tokens)
        audit = AuditEntry(
            trajectory_id=trajectory_id,
            decision_step=decision_step,
            token_budget=self.max_input_tokens - self.reserve_output_tokens,
            actual_tokens=actual,
            decisions=decisions,
            # 审计值由最终 view 计算，不能把 LLM 自报数字当作 token 事实。
            tokens_kept=actual,
            tokens_abstracted=sum(
                self.packer.count_tokens(self.store.get_abstraction(d.step_id, trajectory_id).summary)
                for d in decisions
                if d.elastic_type == ElasticType.ABSTRACT
                and self.store.get_abstraction(d.step_id, trajectory_id) is not None
            ),
            tokens_dropped=sum(
                self.packer.count_tokens("\n".join(s.raw_messages))
                for s in steps
                if not any(d.step_id == s.step_id and d.elastic_type != ElasticType.DROP for d in decisions)
            ),
            fallback_reasons=[d.reason for d in decisions if d.fallback],
            final_message_hash=_hash_messages(messages),
            wall_ms=(time.time() - start) * 1000,
        )
        return messages, audit

    # ------------------------------------------------------------------
    # per-mode builders
    # ------------------------------------------------------------------
    def _build_full(
        self, steps: List[StepRecord], system_prompt: str, tool_schema: str, task: str
    ) -> Tuple[List[Dict[str, str]], List[Decision]]:
        messages: List[Dict[str, str]] = []
        if system_prompt or tool_schema:
            messages.append({"role": "system", "content": "\n".join(p for p in (system_prompt, tool_schema) if p)})
        messages.append({"role": "user", "content": "Task: " + task})
        decisions: List[Decision] = []
        for s in steps:
            for msg in s.raw_messages:
                messages.append({"role": s.role, "content": msg})
            decisions.append(Decision(s.step_id, ElasticType.RAW, "full mode", 1.0))
        return messages, decisions

    def _build_truncate(
        self, steps: List[StepRecord], system_prompt: str, tool_schema: str, task: str
    ) -> Tuple[List[Dict[str, str]], List[Decision]]:
        messages: List[Dict[str, str]] = []
        if system_prompt or tool_schema:
            messages.append({"role": "system", "content": "\n".join(p for p in (system_prompt, tool_schema) if p)})
        messages.append({"role": "user", "content": "Task: " + task})
        decisions: List[Decision] = []
        keep = steps[-self.window_size :] if steps else []
        for s in steps:
            if s in keep:
                for msg in s.raw_messages:
                    messages.append({"role": s.role, "content": msg})
                decisions.append(Decision(s.step_id, ElasticType.RAW, "truncate window", 1.0))
            else:
                decisions.append(Decision(s.step_id, ElasticType.DROP, "outside window", 0.0))
        return messages, decisions

    def _build_summarize(
        self, steps: List[StepRecord], system_prompt: str, tool_schema: str, task: str
    ) -> Tuple[List[Dict[str, str]], List[Decision]]:
        messages: List[Dict[str, str]] = []
        if system_prompt or tool_schema:
            messages.append({"role": "system", "content": "\n".join(p for p in (system_prompt, tool_schema) if p)})
        messages.append({"role": "user", "content": "Task: " + task})
        decisions: List[Decision] = []
        latest = max((s.step_id for s in steps), default=-1)
        old = [s for s in steps if s.step_id != latest]
        summary = self._summary.get(steps[0].trajectory_id) if steps else None
        if old and summary:
            messages.append({"role": "assistant", "content": "[history summary] " + summary})
            decisions.append(Decision(-1, ElasticType.ABSTRACT, "summarize mode rolling summary", 0.5))
        for s in old:
            decisions.append(Decision(s.step_id, ElasticType.DROP, "folded into summary", 0.0))
        for s in steps:
            if s.step_id == latest:
                for msg in s.raw_messages:
                    messages.append({"role": s.role, "content": msg})
                decisions.append(Decision(s.step_id, ElasticType.RAW, "summarize mode latest", 1.0))
        return messages, decisions

    def _build_elastic(
        self, task: str, steps: List[StepRecord], system_prompt: str, tool_schema: str, trajectory_id: str
    ) -> Tuple[List[Dict[str, str]], List[Decision]]:
        latest = max((s.step_id for s in steps), default=-1)
        abstractions: Dict[int, Abstraction] = {
            sid: abs_
            for sid in self.store.steps(trajectory_id)
            if (abs_ := self.store.get_abstraction(sid, trajectory_id)) is not None
        }
        latest_state = " ".join(steps[-1].raw_messages) if steps else ""
        decisions = self.orchestrator.orchestrate(task, latest_state, steps, abstractions)
        messages, _, _ = self.packer.pack(
            task=task,
            steps=steps,
            decisions=decisions,
            abstractions=abstractions,
            system_prompt=system_prompt,
            tool_schema=tool_schema,
            max_input_tokens=self.max_input_tokens,
        )
        return messages, decisions

    # ------------------------------------------------------------------
    # offline replay: rebuild views without calling the environment
    # ------------------------------------------------------------------
    def replay_view(
        self, trajectory: List[StepRecord], task: str, system_prompt: str = ""
    ) -> List[Tuple[int, List[Dict[str, str]], AuditEntry]]:
        """Given a fully recorded trajectory, rebuild the view at each step.

        Deterministic: same input -> same views, no environment calls.
        """
        views: List[Tuple[int, List[Dict[str, str]], AuditEntry]] = []
        policy = ContextPolicy(
            mode=self.mode,
            max_input_tokens=self.max_input_tokens,
            reserve_output_tokens=self.reserve_output_tokens,
            keep_last_raw=self.keep_last_raw,
            llm=self.llm,
            token_counter=self.packer.count_tokens,
            window_size=self.window_size,
        )
        for rec in trajectory:
            policy.record_step(rec.trajectory_id, rec.step_id, rec.role, rec.raw_messages, rec.token_count)
            messages, audit = policy.build(task, system_prompt=system_prompt, decision_step=rec.step_id)
            views.append((rec.step_id, messages, audit))
        return views

    def _roll_summary(self, trajectory_id: str, raw_messages: List[str]) -> None:
        """Cheap rolling summary for `summarize` mode (no model needed)."""
        tail = " ".join(raw_messages)
        previous = self._summary.get(trajectory_id)
        if previous is None:
            self._summary[trajectory_id] = tail[:500]
        else:
            self._summary[trajectory_id] = (previous[:250] + " ... " + tail[:250])[:500]


def _default_counter(text: str) -> int:
    return max(1, len(text) // 4)


def _count_messages(messages: List[Dict[str, str]], counter: Callable[[str], int]) -> int:
    return sum(counter(m.get("content", "")) for m in messages)


def _hash_messages(messages: List[Dict[str, str]]) -> str:
    payload = "\x00".join(f"{m.get('role')}|{m.get('content')}" for m in messages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
