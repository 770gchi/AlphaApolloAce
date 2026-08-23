"""ACE: Adaptive Context Elasticizer — training-free context management.

Packages:
- llm          LLM client abstraction (OpenAI-compatible + mock)
- models       data schemas (StepRecord, Abstraction, Decision, AuditEntry)
- store        A1 lossless message maintenance
- abstraction  A2 per-step structured abstraction + deterministic fallback
- orchestrator A3 per-step elastic orchestration (Raw/Abstract/Drop)
- packing      A4 tokenizer-aware budget packing
- policy       unified ContextPolicy: full | truncate | summarize | elastic
"""
from .abstraction import AbstractionService
from .llm import LLMClient, MockClient, OpenAICompatClient
from .models import Abstraction, AuditEntry, Decision, ElasticType, StepRecord
from .orchestrator import ElasticOrchestrator
from .packing import BudgetPacker
from .policy import ContextPolicy
from .store import LosslessStore
from .runtime import BatchContextRuntime, ContextView, ManagementRequest

__all__ = [
    "Abstraction",
    "AbstractionService",
    "AuditEntry",
    "BudgetPacker",
    "BatchContextRuntime",
    "ContextView",
    "ContextPolicy",
    "Decision",
    "ElasticOrchestrator",
    "ElasticType",
    "LLMClient",
    "LosslessStore",
    "MockClient",
    "OpenAICompatClient",
    "ManagementRequest",
    "StepRecord",
]

__version__ = "0.1.0"
