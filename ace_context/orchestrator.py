"""A3: Per-step elastic orchestration.

At every decision point, the orchestrator assigns each historical step one of
{RAW, ABSTRACT, DROP}. Design rule (from the paper + task): the LLM only
proposes relevance + representation; exact token arithmetic is done by the
packer. The last step is kept RAW by default (last-step invariant).

On ANY LLM failure (bad JSON, timeout, missing fields) we fall back to a
deterministic heuristic (recency + keyword overlap) so behavior stays
reproducible.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .llm import LLMClient
from .models import Abstraction, Decision, ElasticType, StepRecord

ORCHESTRATION_PROMPT = """You decide how to represent each historical step of an agent
trajectory for the CURRENT decision. The final step is handled separately and is
always raw.

Return STRICT JSON, a list of objects, one per step you want to include
(steps 1..{n}):
[
  {{"step_id": 1, "relevance": 0.8, "representation": "raw"}},
  {{"step_id": 2, "relevance": 0.2, "representation": "abstract"}}
]
- relevance: float in [0,1], how critical this step is to the current goal.
- representation: "raw" when exact details matter, "abstract" when the gist
  suffices.
- Steps you omit from the list will be dropped.
Do not output anything else.

Current goal: {task}

Latest state (step {latest}):
{state}

Historical steps (summaries only):
{steps_summary}
"""


class OrchestrationError(Exception):
    pass


class ElasticOrchestrator:
    def __init__(self, llm: Optional[LLMClient] = None, keep_last_raw: bool = True):
        self.llm = llm
        self.keep_last_raw = keep_last_raw

    def orchestrate(
        self,
        task: str,
        latest_state: str,
        steps: List[StepRecord],
        abstractions: Dict[int, Abstraction],
    ) -> List[Decision]:
        if not steps:
            return []
        latest = max(s.step_id for s in steps)
        past = [s for s in steps if s.step_id != latest]
        decisions: List[Decision] = []

        if self.llm is None:
            return self._heuristic(task, latest_state, past, abstractions, latest)

        proposed = None
        try:
            step_summary = "\n".join(
                "step {sid} [{role}] (hash {h}): {abs}".format(
                    sid=s.step_id,
                    role=s.role,
                    h=s.source_hash,
                    abs=(
                        abstractions[s.step_id].summary
                        if s.step_id in abstractions
                        else "NO ABSTRACTION"
                    ),
                )
                for s in past
            )
            prompt = ORCHESTRATION_PROMPT.format(
                task=(task or "")[:2000],
                latest=latest,
                state=(latest_state or "")[:2000],
                steps_summary=step_summary[:8000],
                n=max(latest - 1, 0),
            )
            out = self.llm.complete(prompt, max_tokens=1024, temperature=0.0)
            proposed = _parse_proposals(out)
            valid_ids = {s.step_id for s in past}
            # LLM 只能给内部历史步作建议；越界 id 代表协议不可信，整体回退。
            if set(proposed) - valid_ids:
                raise OrchestrationError("proposal contains an unknown step_id")
        except Exception:  # noqa: BLE001
            proposed = None

        if proposed is None:
            decisions = self._heuristic(task, latest_state, past, abstractions, latest)
            for d in decisions:
                d.fallback = True
        else:
            for s in past:
                prop = proposed.get(s.step_id)
                if prop is None:
                    decisions.append(
                        Decision(s.step_id, ElasticType.DROP, "not proposed by orchestrator", 0.0)
                    )
                else:
                    etype = (
                        ElasticType.RAW
                        if prop["representation"] == "raw"
                        else ElasticType.ABSTRACT
                    )
                    decisions.append(
                        Decision(
                            s.step_id,
                            etype,
                            f"relevance={prop['relevance']:.2f}",
                            prop["relevance"],
                        )
                    )

        # 默认强制 RAW；关闭时保留为显式消融，而不是悄悄忽略配置。
        if self.keep_last_raw:
            decisions.append(Decision(latest, ElasticType.RAW, "last-step invariant", 1.0))
        else:
            decisions.append(Decision(latest, ElasticType.DROP, "last-step ablation: keep_last_raw=false", 0.0))
        return decisions

    # ------------------------------------------------------------------
    def _heuristic(
        self,
        task: str,
        latest_state: str,
        past: List[StepRecord],
        abstractions: Dict[int, Abstraction],
        latest: int,
    ) -> List[Decision]:
        """Deterministic fallback: recency + keyword overlap with the task."""
        task_tokens = _tokenize(task.lower())
        decisions: List[Decision] = []
        for s in past:
            raw_text = " ".join(s.raw_messages).lower()
            overlap = len(task_tokens & _tokenize(raw_text)) / max(1, len(task_tokens))
            if overlap >= 0.15:
                decisions.append(
                    Decision(s.step_id, ElasticType.RAW, f"heuristic overlap={overlap:.2f}", overlap)
                )
            elif overlap >= 0.05:
                decisions.append(
                    Decision(s.step_id, ElasticType.ABSTRACT, f"heuristic overlap={overlap:.2f}", overlap)
                )
            else:
                decisions.append(
                    Decision(s.step_id, ElasticType.DROP, "heuristic low overlap", overlap)
                )
        if self.keep_last_raw:
            decisions.append(
                Decision(latest, ElasticType.RAW, "last-step invariant (heuristic)", 1.0)
            )
        else:
            decisions.append(
                Decision(latest, ElasticType.DROP, "last-step ablation (heuristic)", 0.0)
            )
        return decisions


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text))


def _parse_proposals(text: str) -> Dict[int, Dict]:
    """Parse the LLM's JSON list into {step_id: {relevance, representation}}."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            raise OrchestrationError("no JSON array found in LLM output")
        data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise OrchestrationError("expected a JSON list")

    out: Dict[int, Dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item.get("step_id", -1))
        except (TypeError, ValueError):
            continue
        if sid < 0 or sid in out:
            raise OrchestrationError("invalid or duplicate step_id")
        rep = str(item.get("representation", "abstract")).lower()
        if rep not in ("raw", "abstract"):
            raise OrchestrationError("unknown representation")
        try:
            rel = float(item.get("relevance", 0.0))
        except (TypeError, ValueError):
            rel = 0.0
        out[sid] = {"relevance": max(0.0, min(1.0, rel)), "representation": rep}
    return out
