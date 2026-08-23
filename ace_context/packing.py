"""A4：预算装箱。

选择优先级只决定「哪些完整 step 能入选」；最终消息仍按 step_id 时间顺序输出，
避免 assistant/tool 消息被拆开或重排后破坏对话语义。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .models import Abstraction, Decision, ElasticType, StepRecord


class BudgetExceededError(RuntimeError):
    """P0 内容本身超预算时显式失败，禁止静默丢失最新状态。"""


def _default_counter(text: str) -> int:
    """离线测试的回退；生产环境必须传入模型实际 tokenizer。"""
    return max(1, len(text) // 4)


def _render_abstraction(abs_: Abstraction) -> str:
    """结构化字段也必须进入主模型上下文，不能只依赖 summary。"""
    parts = [f"[ACE abstraction, source step {abs_.source_step}]", abs_.summary]
    if abs_.facts:
        parts.append("Facts: " + "; ".join(abs_.facts))
    if abs_.variables:
        parts.append("Variables: " + "; ".join(f"{k}={v}" for k, v in abs_.variables.items()))
    if abs_.tool_calls:
        parts.append("Tool evidence: " + "; ".join(str(call) for call in abs_.tool_calls))
    if abs_.open_subgoals:
        parts.append("Open subgoals: " + "; ".join(abs_.open_subgoals))
    return "\n".join(part for part in parts if part)


class BudgetPacker:
    def __init__(self, count_tokens: Callable[[str], int] = _default_counter, reserve_output_tokens: int = 512):
        self.count_tokens = count_tokens
        self.reserve_output_tokens = reserve_output_tokens

    def pack(self, task: str, steps: List[StepRecord], decisions: List[Decision], abstractions: Dict[int, Abstraction], system_prompt: str = "", tool_schema: str = "", max_input_tokens: int = 4096) -> Tuple[List[Dict[str, str]], int, int]:
        """在预算内选择完整 step；最新 raw 放不下时 deterministic fail-closed。"""
        budget = max_input_tokens - self.reserve_output_tokens
        if budget < 0:
            raise BudgetExceededError("reserve_output_tokens exceeds max_input_tokens")
        messages = []
        if system_prompt or tool_schema:
            messages.append({"role": "system", "content": "\n".join(p for p in (system_prompt, tool_schema) if p)})
        messages.append({"role": "user", "content": "Task: " + task})
        used = sum(self.count_tokens(m["content"]) for m in messages)
        if used > budget:
            raise BudgetExceededError("system prompt and current task exceed context budget")

        by_id = {s.step_id: s for s in steps}
        dec_by_id = {d.step_id: d for d in decisions if d.step_id in by_id}
        latest = max(by_id, default=-1)
        candidates = []
        for sid, rec in by_id.items():
            d = dec_by_id.get(sid, Decision(sid, ElasticType.DROP))
            if d.elastic_type == ElasticType.RAW:
                content = "\n".join(rec.raw_messages)
            elif d.elastic_type == ElasticType.ABSTRACT and sid in abstractions:
                content = _render_abstraction(abstractions[sid])
            else:
                continue
            candidates.append(((0 if sid == latest else 1 if d.elastic_type == ElasticType.RAW else 2, -d.score, sid), sid, rec, content))

        selected = {}
        for _, sid, rec, content in sorted(candidates):
            size = self.count_tokens(content)
            if used + size <= budget:
                selected[sid] = (rec, content)
                used += size
            elif sid == latest and dec_by_id.get(sid, Decision(sid, ElasticType.DROP)).elastic_type == ElasticType.RAW:
                raise BudgetExceededError("latest raw step exceeds remaining context budget")

        # 选择完成后恢复时间顺序；一个 step 只会作为一个原子消息写入。
        for sid in sorted(selected):
            rec, content = selected[sid]
            messages.append({"role": rec.role, "content": content})
        dropped = sum(self.count_tokens("\n".join(rec.raw_messages)) for sid, rec in by_id.items() if sid not in selected)
        return messages, used, dropped
