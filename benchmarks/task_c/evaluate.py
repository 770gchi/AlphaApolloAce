"""Aggregate Task-C correctness, recovery, cost, and context timelines."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def read_cases(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return {case["case_id"]: case for line in stream if line.strip() for case in [json.loads(line)]}


def _case_id(row: Dict[str, Any]) -> str:
    extra = row.get("extra_info", {})
    if isinstance(extra, str):
        extra = json.loads(extra)
    if isinstance(extra, dict) and extra.get("case_id"):
        return str(extra["case_id"])
    env = row.get("env_kwargs", {})
    return str(env.get("case", {}).get("case_id", "")) if isinstance(env, dict) else ""


def _sample_records(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = row.get("trajectory_records", [])
    if isinstance(records, str):
        records = json.loads(records)
    if records and isinstance(records[0], list):
        return records[0]
    return records or []


def _decision_type(audit: Dict[str, Any], step_id: int) -> str:
    for decision in audit.get("decisions", []):
        if int(decision.get("step_id", -1)) == int(step_id):
            value = decision.get("elastic_type", "drop")
            return str(getattr(value, "value", value)).lower()
    return "drop"


def _abstract_sufficient(case: Dict[str, Any], audit: Dict[str, Any]) -> bool:
    view = str(audit.get("final_observation", ""))
    required = case["required_evidence"]
    if case["category"] == "tool-error-recovery":
        keys = ("tool", "input", "error_code")
    elif case["category"] == "source-sensitive":
        keys = ("source", "exact_value")
    else:
        keys = tuple(key for key in ("source", "exact_value") if key in required)
    return all(str(required[key]) in view for key in keys)


def score_case(case: Dict[str, Any], records: List[Dict[str, Any]], policy: str, budget: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    policy_mode = policy.split("_", 1)[0]
    final_response = str(records[-1].get("response", "")) if records else ""
    correct = re.fullmatch(case["answer_regex"], final_response.strip(), flags=re.IGNORECASE) is not None
    evidence_ids = [int(value) for value in case["evidence_step_ids"]]
    switch = int(case["target_switch_step"])
    eligible_ids, recovered_ids, raw_recovered, abstract_recovered = set(), set(), set(), set()
    per_step = []
    management_tokens = 0
    fallback_count = 0
    type_counts = {"raw": 0, "abstract": 0, "drop": 0}
    for record in records:
        audit = record.get("context_audit") or {}
        decision_step = int(audit.get("decision_step", record.get("step_id", 0)))
        management_tokens += int(audit.get("management_input_tokens", 0)) + int(audit.get("management_output_tokens", 0))
        fallback_count += int(bool(audit.get("fallback_reasons")))
        for decision in audit.get("decisions", []):
            kind = str(decision.get("elastic_type", "drop")).lower()
            type_counts[kind] = type_counts.get(kind, 0) + 1
            per_step.append({
                "case_id": case["case_id"], "category": case["category"], "length_tier": case["length_tier"],
                "policy": policy, "budget": budget, "decision_step": decision_step,
                "history_step": int(decision["step_id"]), "elastic_type": kind,
                "score": float(decision.get("score", 0.0)), "reason": decision.get("reason", ""),
                "prompt_tokens": int(record.get("prompt_tokens", 0)),
            })
        for evidence_id in evidence_ids:
            kind = _decision_type(audit, evidence_id) if audit else ("raw" if policy_mode == "full" else "drop")
            if decision_step < switch and kind in {"drop", "abstract"}:
                eligible_ids.add(evidence_id)
            if decision_step >= switch and evidence_id in eligible_ids:
                if kind == "raw":
                    recovered_ids.add(evidence_id); raw_recovered.add(evidence_id)
                elif kind == "abstract" and _abstract_sufficient(case, audit):
                    recovered_ids.add(evidence_id); abstract_recovered.add(evidence_id)
    eligible = len(eligible_ids)
    recovered = len(recovered_ids)
    total_types = sum(type_counts.values()) or 1
    superseded_value = str(case.get("required_evidence", {}).get("superseded_value", ""))
    erroneous_recovery = bool(superseded_value and superseded_value in final_response)
    prompt_values = [int(record.get("prompt_tokens", 0)) for record in records]
    truncation_events = [record.get("truncation_telemetry") or {} for record in records]
    truncation_count = sum(int(bool(event.get("was_truncated"))) for event in truncation_events)
    total_truncated_tokens = sum(int(event.get("truncated_tokens", 0)) for event in truncation_events)
    pre_truncation_values = [int(event.get("pre_truncation_tokens", 0)) for event in truncation_events]
    post_truncation_values = [int(event.get("post_truncation_tokens", 0)) for event in truncation_events]
    truncation_directions = sorted({str(event.get("truncation_direction", "none")) for event in truncation_events if event.get("was_truncated")})
    result = {
        "case_id": case["case_id"], "category": case["category"], "length_tier": case["length_tier"],
        "policy": policy, "budget": budget, "reference_full_prompt_tokens": case["reference_full_prompt_tokens"],
        "correct": int(correct), "eligible": eligible, "recovered": recovered,
        "recovery_rate": recovered / eligible if eligible else None,
        "raw_recovery_rate": len(raw_recovered) / eligible if eligible else None,
        "abstract_sufficient_rate": len(abstract_recovered) / eligible if eligible else None,
        "erroneous_recovery": int(erroneous_recovery), "final_response": final_response,
        "mean_prompt_tokens": sum(prompt_values) / len(prompt_values) if prompt_values else 0,
        "max_prompt_tokens": max(prompt_values, default=0), "main_generation_tokens": sum(int(r.get("response_tokens", 0)) for r in records),
        "management_tokens": management_tokens, "fallback_steps": fallback_count,
        "raw_ratio": type_counts["raw"] / total_types, "abstract_ratio": type_counts["abstract"] / total_types,
        "drop_ratio": type_counts["drop"] / total_types,
        "max_pre_truncation_tokens": max(pre_truncation_values, default=0),
        "max_post_truncation_tokens": max(post_truncation_values, default=0),
        "truncated_tokens": total_truncated_tokens,
        "truncation_count": truncation_count,
        "truncation_direction": "+".join(truncation_directions) if truncation_directions else "none",
        "context_overflow_or_truncation": truncation_count,
        "failure": int(not records),
    }
    return result, per_step


def evaluate(cases_path: str, output_path: str, policy: str, budget: int, output_dir: str) -> None:
    cases = read_cases(cases_path)
    frame = pd.read_json(output_path, lines=True) if output_path.endswith((".json", ".jsonl")) else pd.read_parquet(output_path)
    case_rows, step_rows = [], []
    for row in frame.to_dict(orient="records"):
        case_id = _case_id(row)
        if case_id not in cases:
            raise ValueError(f"output row has unknown case_id {case_id!r}")
        result, steps = score_case(cases[case_id], _sample_records(row), policy, budget)
        case_rows.append(result); step_rows.extend(steps)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(case_rows)
    per_step = pd.DataFrame(step_rows)
    per_case.to_csv(out / "per_case.csv", index=False)
    per_step.to_json(out / "per_step.jsonl", orient="records", lines=True, force_ascii=False)
    group_cols = ["policy", "budget", "length_tier", "category"]
    summary = per_case.groupby(group_cols, dropna=False).agg(
        cases=("case_id", "count"), accuracy=("correct", "mean"), eligible=("eligible", "sum"), recovered=("recovered", "sum"),
        mean_prompt_tokens=("mean_prompt_tokens", "mean"), management_tokens=("management_tokens", "sum"),
        fallback_steps=("fallback_steps", "sum"), truncations=("truncation_count", "sum"),
        truncated_tokens=("truncated_tokens", "sum"), max_pre_truncation_tokens=("max_pre_truncation_tokens", "max"),
        failures=("failure", "sum"),
    ).reset_index()
    summary["recovery_rate"] = summary["recovered"] / summary["eligible"].where(summary["eligible"] > 0)
    summary.to_csv(out / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--policy", required=True); parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); evaluate(args.cases, args.output, args.policy, args.budget, args.output_dir)


if __name__ == "__main__": main()
