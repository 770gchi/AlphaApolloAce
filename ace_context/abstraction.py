"""A2: Per-step abstraction service with deterministic fallback.

The LLM produces a structured abstraction (summary / facts / variables /
tool_calls / open_subgoals). On ANY failure (invalid JSON, timeout, missing
fields) we fall back to a rule-based abstraction that needs no model, so the
whole pipeline stays runnable offline.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm import LLMClient
from .models import Abstraction

ABSTRACTION_PROMPT = """You are compressing one step of an agent trajectory.
Preserve: exact numbers, variable names, constraints, confirmed facts,
tool-call inputs/outputs/errors, and unresolved sub-goals.
Output STRICT JSON with EXACTLY these keys:
{{
  "summary": "one or two sentence natural language summary",
  "facts": ["confirmed facts as short strings"],
  "variables": {{"name": "value"}},
  "tool_calls": [{{"name": "tool_name", "input": "...", "output": "...", "error": "..."}}],
  "open_subgoals": ["unresolved sub-goals"]
}}
Do not output anything else.

Step {step_id} ({role}):
{content}
"""


class AbstractionError(Exception):
    pass


class AbstractionService:
    def __init__(self, llm: Optional[LLMClient] = None, max_abstraction_tokens: int = 256):
        self.llm = llm
        self.max_abstraction_tokens = max_abstraction_tokens

    def abstract(self, step_id: int, role: str, raw_messages: List[str]) -> Abstraction:
        content = "\n".join(raw_messages)
        if self.llm is None:
            return self._deterministic_fallback(step_id, content)

        try:
            out = self.llm.complete(
                ABSTRACTION_PROMPT.format(
                    step_id=step_id, role=role, content=content[:8000]
                ),
                max_tokens=self.max_abstraction_tokens,
                temperature=0.0,
            )
            parsed = _parse_json(out)
            return Abstraction(
                summary=(str(parsed.get("summary", "")).strip() or _truncate(content))[:500],
                facts=[str(f)[:200] for f in _as_list(parsed.get("facts"))][:20],
                variables={
                    str(k)[:100]: str(v)[:200] for k, v in _as_dict(parsed.get("variables")).items()
                },
                tool_calls=_normalize_tool_calls(parsed.get("tool_calls")),
                open_subgoals=[str(s)[:200] for s in _as_list(parsed.get("open_subgoals"))][:20],
                source_step=step_id,
                generator="llm",
            )
        except Exception as exc:  # noqa: BLE001 — any failure falls back
            abs_ = self._deterministic_fallback(step_id, content)
            abs_.generator = f"fallback:{type(exc).__name__}"
            return abs_

    def _deterministic_fallback(self, step_id: int, content: str) -> Abstraction:
        """Rule-based abstraction: structured extraction, no model required."""
        return Abstraction(
            summary=_truncate(content, 300),
            facts=[ln.strip() for ln in content.splitlines() if ln.strip()][:10],
            variables=_extract_variables(content),
            tool_calls=_extract_tool_calls(content),
            open_subgoals=_extract_subgoals(content),
            source_step=step_id,
            generator="fallback:deterministic",
        )


# --------------------------------------------------------------------------
# parsing helpers (shared by abstraction + orchestration)
# --------------------------------------------------------------------------

def _parse_json(text: str) -> Dict[str, Any]:
    """Robust JSON parse: strip ``` fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise AbstractionError("no JSON object found in LLM output")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise AbstractionError("expected a JSON object")
    return obj


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return [] if value is None else [value]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_tool_calls(raw: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in _as_list(raw)[:10]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": str(item.get("name", "unknown"))[:100],
                "input": str(item.get("input", ""))[:500],
                "output": str(item.get("output", ""))[:500],
                "error": str(item.get("error", ""))[:200],
            }
        )
    return out


def _extract_tool_calls(content: str) -> List[Dict[str, str]]:
    calls: List[Dict[str, str]] = []
    for m in re.finditer(r"(?:tool|function|call)[:\s]*(\w+)[^\n]*(?:\n|$)", content, re.I):
        calls.append({"name": m.group(1)[:100], "input": "", "output": "", "error": ""})
    return calls[:10]


def _extract_variables(content: str) -> Dict[str, str]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", content)
    return {"found_numbers": ",".join(nums[:20])} if nums else {}


def _extract_subgoals(content: str) -> List[str]:
    return [
        ln.strip()[:200]
        for ln in content.splitlines()
        if re.search(r"(still need|remaining|next|todo|unresolved)", ln, re.I)
    ][:20]


def _truncate(text: str, n: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + "...[truncated]"
