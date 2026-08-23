"""ACE core data models (A1 schemas).

These dataclasses are the single source of truth for what the store keeps,
what the orchestrator decides, and what the audit log records.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2


class ElasticType(str, Enum):
    RAW = "raw"
    ABSTRACT = "abstract"
    DROP = "drop"

    def __str__(self) -> str:
        return self.value


def content_hash(text: str) -> str:
    """Stable content hash used for abstraction cache invalidation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Abstraction:
    """Structured compression of a single trajectory step (A2 output)."""

    summary: str
    facts: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    tool_calls: List[Dict[str, str]] = field(default_factory=list)
    open_subgoals: List[str] = field(default_factory=list)
    source_step: int = -1
    source_hash: str = ""
    generator: str = "unknown"  # "llm" | "fallback:deterministic" | ...
    version: str = "1"
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Abstraction":
        return cls(**d)


@dataclass
class StepRecord:
    """Immutable record of one trajectory step (A1). Raw messages never change."""

    trajectory_id: str
    step_id: int
    role: str  # "system" | "user" | "assistant" | "tool"
    raw_messages: List[str]
    message_parts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    abstraction: Optional[Abstraction] = None
    source_hash: str = ""
    timestamp: float = 0.0
    schema_version: int = SCHEMA_VERSION
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.source_hash:
            self.source_hash = content_hash("".join(self.raw_messages))
        if not self.timestamp:
            self.timestamp = time.time()

    def get_raw(self) -> List[str]:
        """Return a defensive copy — raw messages are immutable."""
        return list(self.raw_messages)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["abstraction"] = self.abstraction.to_dict() if self.abstraction else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepRecord":
        d = dict(d)
        if d.get("abstraction"):
            d["abstraction"] = Abstraction.from_dict(d["abstraction"])
        return cls(**d)


@dataclass
class Decision:
    """One elastic decision for one step at one decision point (A3)."""

    step_id: int
    elastic_type: ElasticType
    reason: str = ""
    score: float = 0.0
    fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditEntry:
    """Per-decision audit log (Task B requirement)."""

    trajectory_id: str
    decision_step: int
    token_budget: int
    actual_tokens: int
    decisions: List[Decision] = field(default_factory=list)
    tokens_kept: int = 0
    tokens_abstracted: int = 0
    tokens_dropped: int = 0
    abstraction_cache_hits: int = 0
    fallback_reasons: List[str] = field(default_factory=list)
    final_message_hash: str = ""
    wall_ms: float = 0.0
    mode: str = "elastic"
    reserve_output_tokens: int = 0
    fixed_prompt_tokens: int = 0
    management_input_tokens: int = 0
    management_output_tokens: int = 0
    abstraction_calls: int = 0
    orchestration_calls: int = 0
    management_wall_ms: float = 0.0
    source_hashes: Dict[str, str] = field(default_factory=dict)
    final_observation: str = ""
    final_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
