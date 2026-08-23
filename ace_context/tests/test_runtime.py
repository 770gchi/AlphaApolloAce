from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ace_context import BatchContextRuntime, ElasticType
from ace_context.replay import replay


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


def renderer(task):
    def render(history, step_count):
        return f"TASK={task}\nSTEPS={step_count}\nHISTORY:\n{history}\nANSWER:"
    return render


def test_batch_isolation_and_budget():
    runtime = BatchContextRuntime(mode="truncate", max_input_tokens=500, reserve_output_tokens=50, truncate_window_steps=1)
    runtime.record_interaction("a", 1, "action-a", "observation-a", {"task": "A"})
    runtime.record_interaction("b", 1, "action-b", "observation-b", {"task": "B"})
    views = runtime.build_views(
        ["a", "b"],
        {"a": "A", "b": "B"},
        {"a": renderer("A"), "b": renderer("B")},
        TinyTokenizer(),
        500,
    )
    assert "action-a" in views["a"].observation and "action-b" not in views["a"].observation
    assert "action-b" in views["b"].observation and "action-a" not in views["b"].observation
    assert all(view.token_count <= 450 for view in views.values())
    assert all(view.audit.decisions[-1].elastic_type == ElasticType.RAW for view in views.values())


def test_management_cache_and_invalid_json_fallback():
    runtime = BatchContextRuntime(mode="elastic", orchestration="llm", max_input_tokens=1000, reserve_output_tokens=100)
    runtime.record_interaction("t", 1, "use 42", "tool returned 42", {"task": "find 42"})
    abs_requests = runtime.prepare_management_requests(["t"], {"t": "find 42"}, "abstraction")
    assert len(abs_requests) == 1
    runtime.apply_management_results(abs_requests, ["not-json"])
    assert runtime.store.get_abstraction(1, "t").generator.startswith("fallback")
    assert runtime.prepare_management_requests(["t"], {"t": "find 42"}, "abstraction") == []
    orch_requests = runtime.prepare_management_requests(["t"], {"t": "find 42"}, "orchestration")
    runtime.apply_management_results(orch_requests, ["invalid"])
    view = runtime.build_views(["t"], {"t": "find 42"}, {"t": renderer("find 42")}, TinyTokenizer(), 1000)["t"]
    assert view.audit.decisions[-1].elastic_type == ElasticType.RAW
    assert view.audit.fallback_reasons


def test_store_audit_and_replay_integrity():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = BatchContextRuntime(
            mode="truncate",
            max_input_tokens=600,
            reserve_output_tokens=50,
            store_dir=f"{tmp}/store",
            audit_dir=f"{tmp}/audit",
        )
        runtime.record_interaction("t", 1, "action", "observation", {"task": "task"})
        view = runtime.build_views(["t"], {"t": "task"}, {"t": renderer("task")}, TinyTokenizer(), 600)["t"]
        output = f"{tmp}/views.jsonl"
        assert replay(f"{tmp}/store/trajectories.jsonl", f"{tmp}/audit/context_audit.jsonl", output) == 0
        row = json.loads(Path(output).read_text().strip())
        assert row["message_hash"] == view.audit.final_message_hash
        assert row["observation"] == view.observation


if __name__ == "__main__":
    tests = [
        test_batch_isolation_and_budget,
        test_management_cache_and_invalid_json_fallback,
        test_store_audit_and_replay_integrity,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
