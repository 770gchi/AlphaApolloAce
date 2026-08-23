from __future__ import annotations

import tempfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omegaconf import OmegaConf

from ace_context import BatchContextRuntime, ElasticType
from benchmarks.task_c.build import CATEGORIES, TIERS, WhitespaceTokenizer, build
from benchmarks.task_c.evaluate import score_case
from alphaapollo.core.environments.delayed_evidence.envs import DelayedEvidenceVectorEnv


def test_build_distribution_and_lengths():
    with tempfile.TemporaryDirectory() as tmp:
        cases = build(tmp, "offline", allow_fallback=True)
        assert len(cases) == 30
        assert sum(c["length_tier"] == "short" for c in cases) == 15
        assert sum(c["length_tier"] == "long" for c in cases) == 15
        for tier, (lower, upper, _, _) in TIERS.items():
            subset = [c for c in cases if c["length_tier"] == tier]
            assert {c["category"] for c in subset} == set(CATEGORIES)
            assert all(lower <= c["reference_full_prompt_tokens"] <= upper for c in subset)
        assert (Path(tmp) / "test.parquet").exists()
        review_files = sorted((Path(tmp) / "cases").glob("*.json"))
        assert len(review_files) == 30
        reviewed = [json.loads(path.read_text(encoding="utf-8")) for path in review_files]
        assert {case["case_id"] for case in reviewed} == {case["case_id"] for case in cases}
        manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["review_files"]) == 30


def test_environment_is_deterministic_and_scores_strictly():
    with tempfile.TemporaryDirectory() as tmp:
        case = build(tmp, "offline", allow_fallback=True, case_id="exact_number_01_short")[0]
    cfg = OmegaConf.create({"max_steps": 17})
    env = DelayedEvidenceVectorEnv(env_num=1, env_config=cfg)
    kwargs = [{"case": case}]
    first, _ = env.reset(kwargs)
    trace_a = []
    for item in case["scripted_observations"]:
        obs, reward, done, _ = env.step(["<ack>"], ["<ack>"])
        trace_a.append(obs[0]); assert not done[0]
    _, reward, done, _ = env.step(["<answer>731942</answer>"], ["<answer>731942</answer>"])
    assert done[0] and reward[0] == 1
    env.reset(kwargs); trace_b = [env.step(["anything"], ["anything"])[0][0] for _ in case["scripted_observations"]]
    assert first == [case["initial_goal"]] and trace_a == trace_b
    env.reset(kwargs)
    for _ in case["scripted_observations"]: env.step(["x"], ["x"])
    _, reward, _, _ = env.step(["<answer>731924</answer>"], ["<answer>731924</answer>"])
    assert reward[0] == 0


def test_recovery_denominator_and_sufficient_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        case = build(tmp, "offline", allow_fallback=True, case_id="exact_number_01_short")[0]
    records = [
        {"step_id": 2, "response": "<ack>", "prompt_tokens": 500, "response_tokens": 2, "context_audit": {"decision_step": 2, "decisions": [{"step_id": 1, "elastic_type": "drop"}]}},
        {"step_id": case["target_switch_step"], "response": "<answer>731942</answer>", "prompt_tokens": 1800, "response_tokens": 8, "context_audit": {"decision_step": case["target_switch_step"], "final_observation": "ALPHA 731942", "decisions": [{"step_id": 1, "elastic_type": "raw"}]}},
    ]
    result, _ = score_case(case, records, "elastic", 2048)
    assert result["correct"] == 1 and result["eligible"] == 1 and result["recovered"] == 1 and result["recovery_rate"] == 1


def test_input_budget_and_allowed_types():
    tok = WhitespaceTokenizer()
    runtime = BatchContextRuntime(mode="elastic", max_input_tokens=128, reserve_output_tokens=32, model_context_window=256, orchestration="heuristic", allowed_types=["raw", "drop"])
    runtime.record_interaction("t", 1, "ack", "critical 42")
    view = runtime.build_views(["t"], {"t": "critical"}, {"t": lambda history, n: f"task\n{history}"}, tok, 128)["t"]
    assert view.audit.token_budget == 128
    assert all(d.elastic_type in {ElasticType.RAW, ElasticType.DROP} for d in view.audit.decisions)


if __name__ == "__main__":
    tests = [test_build_distribution_and_lengths, test_environment_is_deterministic_and_scores_strictly, test_recovery_denominator_and_sufficient_recovery, test_input_budget_and_allowed_types]
    for test in tests:
        test(); print(f"{test.__name__}: OK")
