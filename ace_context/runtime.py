"""Batch-aware ACE runtime used at AlphaApollo's generation boundary."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .abstraction import AbstractionService, _as_dict, _as_list, _normalize_tool_calls, _parse_json
from .models import Abstraction, AuditEntry, Decision, ElasticType, StepRecord
from .orchestrator import ElasticOrchestrator, ORCHESTRATION_PROMPT, _parse_proposals
from .packing import BudgetExceededError, _render_abstraction
from .store import LosslessStore


@dataclass(frozen=True)
class ManagementRequest:
    kind: str
    trajectory_id: str
    step_id: int
    prompt: str
    max_tokens: int


@dataclass
class ContextView:
    trajectory_id: str
    observation: str
    prompt_text: str
    token_count: int
    audit: AuditEntry


class BatchContextRuntime:
    """Owns lossless histories while exposing batched management requests.

    It never calls a model itself.  AlphaApollo executes returned requests with
    the already-loaded rollout worker and feeds responses back through
    :meth:`apply_management_results`.
    """

    VALID_MODES = {"full", "truncate", "summarize", "elastic"}
    VALID_ORCHESTRATION = {"llm", "heuristic"}

    def __init__(
        self,
        mode: str = "full",
        max_input_tokens: int = 4096,
        reserve_output_tokens: int = 512,
        keep_last_raw: bool = True,
        abstraction_cache: bool = True,
        orchestration: str = "llm",
        truncate_window_steps: int = 4,
        abstraction_max_tokens: int = 256,
        orchestration_max_tokens: int = 512,
        allowed_types: Optional[Sequence[str]] = None,
        model_context_window: int = 32768,
        audit_dir: Optional[str] = None,
        store_dir: Optional[str] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"invalid context policy mode: {mode}")
        if orchestration not in self.VALID_ORCHESTRATION:
            raise ValueError(f"invalid orchestration mode: {orchestration}")
        if max_input_tokens <= 0 or reserve_output_tokens < 0:
            raise ValueError("token limits must be positive")
        if max_input_tokens + reserve_output_tokens > model_context_window:
            raise ValueError("input budget plus output reserve exceeds model context window")
        if truncate_window_steps < 1:
            raise ValueError("truncate_window_steps must be >= 1")
        self.mode = mode
        self.max_input_tokens = int(max_input_tokens)
        self.reserve_output_tokens = int(reserve_output_tokens)
        self.keep_last_raw = bool(keep_last_raw)
        self.abstraction_cache = bool(abstraction_cache)
        self.orchestration = orchestration
        self.truncate_window_steps = int(truncate_window_steps)
        self.abstraction_max_tokens = int(abstraction_max_tokens)
        self.orchestration_max_tokens = int(orchestration_max_tokens)
        requested_types = allowed_types or ["raw", "abstract", "drop"]
        self.allowed_types = {ElasticType(str(value).lower()) for value in requested_types}
        if ElasticType.RAW not in self.allowed_types:
            raise ValueError("allowed_types must contain raw")
        if mode == "summarize" and ElasticType.ABSTRACT not in self.allowed_types:
            raise ValueError("summarize mode requires abstract in allowed_types")
        self.model_context_window = int(model_context_window)
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.store_dir = Path(store_dir) if store_dir else None
        self.store = LosslessStore()
        self._fallback_abstraction = AbstractionService(llm=None)
        self._fallback_orchestrator = ElasticOrchestrator(llm=None, keep_last_raw=keep_last_raw)
        self._decisions: Dict[str, List[Decision]] = {}
        self._fallback_reasons: Dict[str, List[str]] = {}
        self._stats: Dict[str, Dict[str, float]] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "full"

    def reset(self) -> None:
        self.store = LosslessStore()
        self._decisions.clear()
        self._fallback_reasons.clear()
        self._stats.clear()

    def record_interaction(
        self,
        trajectory_id: str,
        step_id: int,
        action: str,
        observation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw = [str(action or ""), str(observation or "")]
        rec = StepRecord(
            trajectory_id=str(trajectory_id),
            step_id=int(step_id),
            role="interaction",
            raw_messages=raw,
            message_parts=[
                {"role": "assistant", "content": raw[0]},
                {"role": "tool", "content": raw[1]},
            ],
            metadata=dict(metadata or {}),
        )
        self.store.append(rec)
        if self.store_dir:
            self.store.save(self._store_path())

    def prepare_management_requests(
        self,
        trajectory_ids: Iterable[str],
        tasks: Dict[str, str],
        stage: str,
    ) -> List[ManagementRequest]:
        if stage not in {"abstraction", "orchestration"}:
            raise ValueError("stage must be abstraction or orchestration")
        if not self.enabled or self.mode not in {"summarize", "elastic"}:
            return []
        requests: List[ManagementRequest] = []
        for tid in trajectory_ids:
            steps = self._steps(tid)
            if stage == "abstraction":
                for rec in steps:
                    cached = self.store.get_abstraction(rec.step_id, tid)
                    if cached is not None and self.abstraction_cache:
                        self._stat(tid)["cache_hits"] += 1
                        continue
                    prompt = self._abstraction_prompt(rec)
                    requests.append(ManagementRequest("abstraction", tid, rec.step_id, prompt, self.abstraction_max_tokens))
            elif self.mode == "elastic" and self.orchestration == "llm" and steps:
                prompt = self._orchestration_prompt(tid, tasks.get(tid, ""), steps)
                requests.append(ManagementRequest("orchestration", tid, max(s.step_id for s in steps), prompt, self.orchestration_max_tokens))
        return requests

    def apply_management_results(
        self,
        requests: Sequence[ManagementRequest],
        outputs: Sequence[str],
        input_tokens: Optional[Sequence[int]] = None,
        output_tokens: Optional[Sequence[int]] = None,
        wall_ms: float = 0.0,
    ) -> None:
        if len(requests) != len(outputs):
            raise ValueError("management request/result length mismatch")
        for idx, (req, output) in enumerate(zip(requests, outputs)):
            stats = self._stat(req.trajectory_id)
            stats["management_input_tokens"] += int(input_tokens[idx]) if input_tokens else 0
            stats["management_output_tokens"] += int(output_tokens[idx]) if output_tokens else 0
            stats["management_wall_ms"] += float(wall_ms) / max(1, len(requests))
            if req.kind == "abstraction":
                stats["abstraction_calls"] += 1
                rec = self.store.get(req.step_id, req.trajectory_id)
                if rec is None:
                    continue
                try:
                    obj = _parse_json(output)
                    required = {"summary", "facts", "variables", "tool_calls", "open_subgoals"}
                    if not required.issubset(obj):
                        raise ValueError("abstraction response is missing required fields")
                    abstraction = Abstraction(
                        summary=str(obj.get("summary", "")).strip(),
                        facts=[str(x) for x in _as_list(obj.get("facts"))],
                        variables={str(k): str(v) for k, v in _as_dict(obj.get("variables")).items()},
                        tool_calls=_normalize_tool_calls(obj.get("tool_calls")),
                        open_subgoals=[str(x) for x in _as_list(obj.get("open_subgoals"))],
                        generator="llm",
                    )
                    if not abstraction.summary:
                        raise ValueError("empty abstraction summary")
                except Exception as exc:
                    abstraction = self._fallback_abstraction.abstract(rec.step_id, rec.role, rec.raw_messages)
                    self._fallback(req.trajectory_id, f"abstraction:{type(exc).__name__}")
                self.store.put_abstraction(req.step_id, req.trajectory_id, abstraction)
            else:
                stats["orchestration_calls"] += 1
                try:
                    proposals = _parse_proposals(output)
                    self._decisions[req.trajectory_id] = self._validate_proposals(req.trajectory_id, proposals)
                except Exception as exc:
                    self._decisions[req.trajectory_id] = self._heuristic(req.trajectory_id, "")
                    for decision in self._decisions[req.trajectory_id]:
                        decision.fallback = True
                    self._fallback(req.trajectory_id, f"orchestration:{type(exc).__name__}")
        if self.store_dir and requests:
            self.store.save(self._store_path())

    def apply_management_failure(self, requests: Sequence[ManagementRequest], reason: str) -> None:
        """Deterministically close a failed worker management phase."""
        for req in requests:
            rec = self.store.get(req.step_id, req.trajectory_id)
            if req.kind == "abstraction" and rec is not None:
                fallback = self._fallback_abstraction.abstract(rec.step_id, rec.role, rec.raw_messages)
                self.store.put_abstraction(rec.step_id, req.trajectory_id, fallback)
            elif req.kind == "orchestration":
                self._decisions[req.trajectory_id] = self._heuristic(req.trajectory_id, "")
            self._fallback(req.trajectory_id, f"management:{reason}")

    def build_views(
        self,
        trajectory_ids: Sequence[str],
        tasks: Dict[str, str],
        prompt_renderers: Dict[str, Callable[[str, int], str]],
        tokenizer: Any,
        max_prompt_length: int,
    ) -> Dict[str, ContextView]:
        views: Dict[str, ContextView] = {}
        budget = min(int(max_prompt_length), self.max_input_tokens)
        for tid in trajectory_ids:
            start = time.perf_counter()
            steps = self._steps(tid)
            renderer = prompt_renderers[tid]
            fixed_observation = renderer("", len(steps))
            fixed_prompt = self._chat_prompt(tokenizer, fixed_observation)
            fixed_tokens = len(tokenizer.encode(fixed_prompt, add_special_tokens=False))
            if fixed_tokens > budget:
                raise BudgetExceededError(f"fixed prompt exceeds ACE budget for trajectory {tid}")
            decisions = self._desired_decisions(tid, tasks.get(tid, ""), steps)
            decisions = self._pack(tid, steps, decisions, renderer, tokenizer, budget)
            history = self._render_history(tid, steps, decisions)
            observation = renderer(history, len(steps))
            prompt_text = self._chat_prompt(tokenizer, observation)
            actual_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            if actual_tokens > budget:
                raise BudgetExceededError(f"packed prompt exceeds ACE budget for trajectory {tid}")
            audit = self._make_audit(tid, steps, decisions, fixed_tokens, actual_tokens, observation, prompt_text, tokenizer, budget, start)
            views[tid] = ContextView(tid, observation, prompt_text, actual_tokens, audit)
            self._write_audit(audit)
            self._stats.pop(tid, None)
        return views

    def _desired_decisions(self, tid: str, task: str, steps: List[StepRecord]) -> List[Decision]:
        if not steps:
            return []
        latest = steps[-1].step_id
        if self.mode == "truncate":
            keep = {s.step_id for s in steps[-self.truncate_window_steps:]}
            return [Decision(s.step_id, ElasticType.RAW if s.step_id in keep else ElasticType.DROP, "truncate window", 1.0 if s.step_id in keep else 0.0) for s in steps]
        if self.mode == "summarize":
            return [Decision(s.step_id, ElasticType.RAW if s.step_id == latest else ElasticType.ABSTRACT, "latest raw" if s.step_id == latest else "history summary", 1.0 if s.step_id == latest else 0.5) for s in steps]
        decisions = self._decisions.pop(tid, None)
        if decisions is None or self.orchestration == "heuristic":
            decisions = self._heuristic(tid, task)
        return [self._enforce_allowed_type(decision) for decision in decisions]

    def _enforce_allowed_type(self, decision: Decision) -> Decision:
        if decision.elastic_type in self.allowed_types:
            return decision
        if decision.elastic_type == ElasticType.ABSTRACT:
            replacement = ElasticType.RAW
        elif ElasticType.ABSTRACT in self.allowed_types:
            replacement = ElasticType.ABSTRACT
        else:
            replacement = ElasticType.RAW
        return Decision(decision.step_id, replacement, f"{decision.reason}; {decision.elastic_type.value} disabled -> {replacement.value}", decision.score, decision.fallback)

    def _pack(self, tid: str, steps: List[StepRecord], decisions: List[Decision], renderer: Callable[[str, int], str], tokenizer: Any, budget: int) -> List[Decision]:
        by_id = {s.step_id: s for s in steps}
        desired = {d.step_id: d for d in decisions}
        latest = steps[-1].step_id if steps else -1
        if self.keep_last_raw and latest >= 0:
            desired[latest] = Decision(latest, ElasticType.RAW, "last-step invariant", 1.0)
        candidates = [d for d in desired.values() if d.elastic_type != ElasticType.DROP]
        candidates.sort(key=lambda d: (0 if d.step_id == latest else 1 if d.elastic_type == ElasticType.RAW else 2, -d.score, d.step_id))
        selected: Dict[int, Decision] = {}
        for decision in candidates:
            trial = dict(selected)
            trial[decision.step_id] = decision
            trial_decisions = [trial.get(s.step_id, Decision(s.step_id, ElasticType.DROP, "budget drop")) for s in steps]
            history = self._render_history(tid, steps, trial_decisions)
            prompt = self._chat_prompt(tokenizer, renderer(history, len(steps)))
            size = len(tokenizer.encode(prompt, add_special_tokens=False))
            if size <= budget:
                selected[decision.step_id] = decision
            elif decision.step_id == latest and decision.elastic_type == ElasticType.RAW:
                raise BudgetExceededError(f"latest raw step exceeds ACE budget for trajectory {tid}")
        packed = []
        for step in steps:
            if step.step_id in selected:
                packed.append(selected[step.step_id])
            elif ElasticType.DROP in self.allowed_types:
                score = desired.get(step.step_id, Decision(step.step_id, ElasticType.DROP)).score
                packed.append(Decision(step.step_id, ElasticType.DROP, "budget drop", score))
            else:
                raise BudgetExceededError("context cannot fit when DROP is disabled")
        return packed

    def _render_history(self, tid: str, steps: List[StepRecord], decisions: List[Decision]) -> str:
        by_decision = {d.step_id: d for d in decisions}
        chunks: List[str] = []
        for rec in steps:
            decision = by_decision.get(rec.step_id)
            if decision is None or decision.elastic_type == ElasticType.DROP:
                continue
            if decision.elastic_type == ElasticType.ABSTRACT:
                abstraction = self.store.get_abstraction(rec.step_id, tid)
                if abstraction is None:
                    continue
                chunks.append(_render_abstraction(abstraction))
            else:
                action, observation = (rec.raw_messages + ["", ""])[:2]
                chunks.append(f"[Action {rec.step_id}: '{action}', Observation {rec.step_id}: '{observation}']")
        return "\n".join(chunks)

    def _make_audit(self, tid: str, steps: List[StepRecord], decisions: List[Decision], fixed_tokens: int, actual_tokens: int, observation: str, prompt_text: str, tokenizer: Any, budget: int, started: float) -> AuditEntry:
        stats = self._stat(tid)
        by_id = {s.step_id: s for s in steps}

        def count(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        raw_sizes = {
            sid: count(f"[Action {sid}: '{rec.raw_messages[0]}', Observation {sid}: '{rec.raw_messages[1]}']")
            for sid, rec in by_id.items()
        }
        kept = sum(raw_sizes[d.step_id] for d in decisions if d.elastic_type == ElasticType.RAW)
        abstracted = sum(
            count(_render_abstraction(abs_))
            for d in decisions
            if d.elastic_type == ElasticType.ABSTRACT
            and (abs_ := self.store.get_abstraction(d.step_id, tid)) is not None
        )
        dropped = sum(raw_sizes[d.step_id] for d in decisions if d.elastic_type == ElasticType.DROP)
        return AuditEntry(
            trajectory_id=tid,
            decision_step=max(by_id, default=0),
            token_budget=budget,
            actual_tokens=actual_tokens,
            decisions=decisions,
            tokens_kept=kept,
            tokens_abstracted=abstracted,
            tokens_dropped=dropped,
            abstraction_cache_hits=int(stats["cache_hits"]),
            fallback_reasons=list(self._fallback_reasons.pop(tid, [])),
            final_message_hash=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            wall_ms=(time.perf_counter() - started) * 1000,
            mode=self.mode,
            reserve_output_tokens=self.reserve_output_tokens,
            fixed_prompt_tokens=fixed_tokens,
            management_input_tokens=int(stats["management_input_tokens"]),
            management_output_tokens=int(stats["management_output_tokens"]),
            abstraction_calls=int(stats["abstraction_calls"]),
            orchestration_calls=int(stats["orchestration_calls"]),
            management_wall_ms=stats["management_wall_ms"],
            source_hashes={str(s.step_id): s.source_hash for s in steps},
            final_observation=observation,
            final_prompt=prompt_text,
        )

    def _validate_proposals(self, tid: str, proposals: Dict[int, Dict[str, Any]]) -> List[Decision]:
        steps = self._steps(tid)
        latest = steps[-1].step_id
        valid = {s.step_id for s in steps[:-1]}
        if set(proposals) - valid:
            raise ValueError("orchestrator returned unknown/latest step")
        decisions = []
        for rec in steps[:-1]:
            item = proposals.get(rec.step_id)
            if item is None:
                decisions.append(Decision(rec.step_id, ElasticType.DROP, "not proposed", 0.0))
            else:
                kind = ElasticType.RAW if item["representation"] == "raw" else ElasticType.ABSTRACT
                decisions.append(Decision(rec.step_id, kind, f"relevance={item['relevance']:.2f}", item["relevance"]))
        decisions.append(Decision(latest, ElasticType.RAW if self.keep_last_raw else ElasticType.DROP, "last-step invariant" if self.keep_last_raw else "last-step ablation", 1.0 if self.keep_last_raw else 0.0))
        return decisions

    def _heuristic(self, tid: str, task: str) -> List[Decision]:
        steps = self._steps(tid)
        abstractions = {s.step_id: a for s in steps if (a := self.store.get_abstraction(s.step_id, tid)) is not None}
        latest_state = " ".join(steps[-1].raw_messages) if steps else ""
        return self._fallback_orchestrator.orchestrate(task, latest_state, steps, abstractions)

    def _steps(self, tid: str) -> List[StepRecord]:
        try:
            return [self.store.get(sid, tid) for sid in self.store.steps(tid)]
        except ValueError:
            return []

    def _abstraction_prompt(self, rec: StepRecord) -> str:
        from .abstraction import ABSTRACTION_PROMPT
        return ABSTRACTION_PROMPT.format(step_id=rec.step_id, role=rec.role, content="\n".join(rec.raw_messages)[:8000])

    def _orchestration_prompt(self, tid: str, task: str, steps: List[StepRecord]) -> str:
        latest = steps[-1]
        summaries = "\n".join(f"step {s.step_id} [{s.role}] (hash {s.source_hash}): {(self.store.get_abstraction(s.step_id, tid).summary if self.store.get_abstraction(s.step_id, tid) else 'NO ABSTRACTION')}" for s in steps[:-1])
        return ORCHESTRATION_PROMPT.format(task=task[:2000], latest=latest.step_id, state=" ".join(latest.raw_messages)[:2000], steps_summary=summaries[:8000], n=max(latest.step_id - 1, 0))

    @staticmethod
    def _chat_prompt(tokenizer: Any, observation: str) -> str:
        return tokenizer.apply_chat_template([{"role": "user", "content": observation}], add_generation_prompt=True, tokenize=False)

    def _stat(self, tid: str) -> Dict[str, float]:
        return self._stats.setdefault(tid, {"cache_hits": 0, "management_input_tokens": 0, "management_output_tokens": 0, "abstraction_calls": 0, "orchestration_calls": 0, "management_wall_ms": 0.0})

    def _fallback(self, tid: str, reason: str) -> None:
        self._fallback_reasons.setdefault(tid, []).append(reason)

    def _store_path(self) -> Path:
        assert self.store_dir is not None
        self.store_dir.mkdir(parents=True, exist_ok=True)
        return self.store_dir / "trajectories.jsonl"

    def _write_audit(self, audit: AuditEntry) -> None:
        if not self.audit_dir:
            return
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        with open(self.audit_dir / "context_audit.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit.to_dict(), ensure_ascii=False, default=str) + "\n")
