"""LLM client abstraction for ACE.

Two implementations:
- OpenAICompatClient: talks to any OpenAI-compatible local endpoint
  (vLLM / SGLang serving Qwen2.5-3B-Instruct, or the Qwen DashScope API).
- MockClient: deterministic canned responses, used by offline unit tests
  so the whole core can be exercised without GPU or network.
"""
from __future__ import annotations

import json
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class LLMClient:
    """Minimal protocol: one-shot chat completion, return the assistant text."""

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    """Client for any OpenAI-compatible endpoint (vLLM, SGLang, DashScope)."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        timeout: float = 120.0,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package not installed")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class MockClient(LLMClient):
    """Deterministic mock. Routes by prompt content so offline tests are stable.

    - abstraction prompt  -> a valid (empty) structured abstraction
    - orchestration prompt -> step 1 proposed as raw, everything else dropped
    - otherwise           -> canned replies from a script queue
    """

    def __init__(
        self,
        script: Optional[List[str]] = None,
        fail_abstraction: bool = False,
        invalid_orchestration: bool = False,
    ):
        self.script = list(script or [])
        self.calls: List[str] = []
        self.fail_abstraction = fail_abstraction
        self.invalid_orchestration = invalid_orchestration

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.calls.append(prompt[:80])
        low = prompt.lower()
        if "compressing one step" in low:
            if self.fail_abstraction:
                raise RuntimeError("mock abstraction failure")
            return json.dumps(
                {
                    "summary": "mock summary of the step",
                    "facts": ["mock fact"],
                    "variables": {"mock_key": "mock_value"},
                    "tool_calls": [],
                    "open_subgoals": [],
                }
            )
        if "relevance" in low:
            if self.invalid_orchestration:
                return "this is definitely not json [ { broken"
            return json.dumps(
                [{"step_id": 1, "relevance": 0.9, "representation": "raw"}]
            )
        if self.script:
            idx = min(len(self.calls) - 1, len(self.script) - 1)
            return self.script[idx]
        return "mock reply"
