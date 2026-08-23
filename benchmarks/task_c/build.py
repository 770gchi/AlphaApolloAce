"""Build the frozen Task-C delayed-evidence benchmark and AlphaApollo dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

CATEGORIES = ("exact-number", "source-sensitive", "correction", "tool-error-recovery", "hard-negative")
TIERS = {"short": (8192, 12288, 10240, 12), "long": (12288, 16384, 14336, 16)}
VALUES = {
    "exact-number": ["731942", "50817364", "946201.375"],
    "source-sensitive": ["482761", "ZX-903184", "17.625"],
    "correction": ["665204", "QF-771903", "84.125"],
    "tool-error-recovery": ["E_CONN_407", "ERR_SCHEMA_219", "TIMEOUT_731"],
    "hard-negative": ["ORCHID-592", "CEDAR-1847", "LANTERN-063"],
}


class WhitespaceTokenizer:
    name_or_path = "whitespace-test-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(range(len(re.findall(r"\w+|[^\w\s]", text))))

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)
        return text + ("\n<assistant>" if add_generation_prompt else "")


def load_tokenizer(model: str, allow_fallback: bool = False):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model)
    except Exception:
        if not allow_fallback:
            raise
        return WhitespaceTokenizer()


def initial_goal(case_id: str) -> str:
    return (
        f"Delayed-evidence case {case_id}. You will receive deterministic archived notices. "
        "Until a notice marked FINAL QUERY appears, reply only with <ack>step-N</ack>. "
        "For FINAL QUERY, wrap the exact requested value between the opening tag <answer> "
        "and closing tag </answer>, with no other text. Preserve exact values and source identity."
    )


def _evidence(category: str, value: str, variant: int) -> Tuple[str, Dict[str, Any], List[int]]:
    if category == "exact-number":
        return f"Archive ALPHA confirms the vault access code is exactly {value}.", {"source": "ALPHA", "exact_value": value}, []
    if category == "source-sensitive":
        source = ("NORTH", "ATLAS", "PRIMARY")[variant]
        return f"Authoritative source {source} records the requested registry value as exactly {value}.", {"source": source, "exact_value": value}, []
    if category == "correction":
        old = ("665240", "QF-771309", "84.215")[variant]
        return f"CORRECTION: supersede the earlier provisional value {old}; the authoritative final value is {value}.", {"exact_value": value, "superseded_value": old}, [1]
    if category == "tool-error-recovery":
        tool = ("fetch_ledger", "schema_check", "remote_solver")[variant]
        argument = ("ledger=ALPHA", "table=claims_v2", "job=731")[variant]
        return f"Tool failure record: tool={tool}; input={argument}; exact error_code={value}. Preserve all three fields.", {"tool": tool, "input": argument, "error_code": value}, []
    key = ("ORCHID", "CEDAR", "LANTERN")[variant]
    return f"Dependency ledger states that the only approved release key is {value}; it belongs to dependency node {key}.", {"source": key, "exact_value": value}, []


def _target_query(category: str, required: Dict[str, Any], evidence_step: int) -> str:
    if category == "tool-error-recovery":
        return f"FINAL QUERY: Return the exact error code from the early tool failure at step {evidence_step}. Ignore later successful tools."
    if category == "correction":
        return f"FINAL QUERY: Return the corrected final value from step {evidence_step}, not the superseded provisional value."
    source = required.get("source", "the authoritative source")
    return f"FINAL QUERY: Return the exact value stated by {source} at step {evidence_step}. Ignore every similar identifier from other sources."


def _distractor(case_id: str, step_id: int, variant: int) -> str:
    a = 700000 + ((step_id * 7919 + variant * 104729) % 299999)
    b = 100000 + ((step_id * 3571 + variant * 65537) % 899999)
    source = ("BETA", "GAMMA", "DELTA", "OPS", "MIRROR")[step_id % 5]
    return (
        f"Distractor report {case_id}-{step_id} from {source}. Shipment reference {a}, cabinet {b}, "
        f"and timestamp 19:{step_id:02d} are unrelated to the authoritative evidence. "
        "This report discusses vault operations, archive validation, source reconciliation, tool diagnostics, "
        "and numeric identifiers, but explicitly does not revise or replace any earlier authoritative record."
    )


def _filler(case_id: str, step_id: int, serial: int) -> str:
    x = 100000 + ((serial * 8191 + step_id * 131) % 899999)
    y = 100000 + ((serial * 6151 + step_id * 197) % 899999)
    return (
        f" Audit line {serial} for report {case_id}-{step_id}: observer MIRROR-{serial % 17} compared ticket {x} "
        f"with shipment {y}. The comparison covered archive access, tool recovery, corrected values, source labels, "
        "and exact-number formatting. It is a non-authoritative operational record and must not be used as final evidence."
    )


def render_reference_prompt(case: Dict[str, Any], tokenizer) -> str:
    chunks = []
    for item in case["scripted_observations"]:
        sid = item["step_id"]
        chunks.append(f"[Action {sid}: '<ack>step-{sid}</ack>', Observation {sid}: '{item['content']}']")
    observation = (
        f"{case['initial_goal']}\n\nPrior interactions ({len(chunks)} steps):\n"
        + "\n".join(chunks)
        + "\n\nRespond to the latest notice now."
    )
    return tokenizer.apply_chat_template([{"role": "user", "content": observation}], add_generation_prompt=True, tokenize=False)


def token_count(case: Dict[str, Any], tokenizer) -> int:
    return len(tokenizer.encode(render_reference_prompt(case, tokenizer), add_special_tokens=False))


def make_case(category: str, tier: str, variant: int, tokenizer) -> Dict[str, Any]:
    lower, upper, target, steps_count = TIERS[tier]
    value = VALUES[category][variant]
    case_id = f"{category.replace('-', '_')}_{variant + 1:02d}_{tier}"
    evidence_text, required, superseded = _evidence(category, value, variant)
    observations = []
    if category == "correction":
        observations.append({"step_id": 1, "source": "PROVISIONAL", "content": f"Provisional record lists {required['superseded_value']}; this may later be corrected.", "tags": ["superseded_evidence"]})
        evidence_step = 2
    else:
        evidence_step = 1
    observations.append({"step_id": evidence_step, "source": required.get("source", "TOOL"), "content": evidence_text, "tags": ["critical_evidence"]})
    for sid in range(evidence_step + 1, steps_count):
        observations.append({"step_id": sid, "source": "DISTRACTOR", "content": _distractor(case_id, sid, variant), "tags": ["distractor"]})
    observations.append({"step_id": steps_count, "source": "CONTROL", "content": _target_query(category, required, evidence_step), "tags": ["target_switch"]})
    case = {
        "schema_version": 1,
        "case_id": case_id,
        "category": category,
        "length_tier": tier,
        "initial_goal": initial_goal(case_id),
        "scripted_observations": observations,
        "target_switch_step": steps_count,
        "evidence_step_ids": [evidence_step],
        "superseded_step_ids": superseded,
        "required_evidence": required,
        "expected_answer": value,
        "answer_regex": rf"<answer>\s*{re.escape(value)}\s*</answer>",
        "max_steps": steps_count + 1,
    }
    serial = 0
    filler_steps = [item for item in observations if "distractor" in item["tags"]]
    base_count = token_count(case, tokenizer)
    sample_size = max(1, len(tokenizer.encode(_filler(case_id, evidence_step + 1, 1), add_special_tokens=False)))
    estimated = max(0, int((target - base_count) / sample_size))
    for serial in range(1, estimated + 1):
        item = filler_steps[(serial - 1) % len(filler_steps)]
        item["content"] += _filler(case_id, item["step_id"], serial)
    while token_count(case, tokenizer) < target:
        serial += 1
        item = filler_steps[(serial - 1) % len(filler_steps)]
        item["content"] += _filler(case_id, item["step_id"], serial)
    count = token_count(case, tokenizer)
    if not lower <= count <= upper:
        raise ValueError(f"{case_id} has {count} tokens, expected [{lower}, {upper}]")
    case["reference_full_prompt_tokens"] = count
    return case


def validate_case(case: Dict[str, Any], tokenizer) -> None:
    required_keys = {"case_id", "category", "length_tier", "scripted_observations", "target_switch_step", "evidence_step_ids", "required_evidence", "expected_answer"}
    missing = required_keys - set(case)
    if missing:
        raise ValueError(f"{case.get('case_id')}: missing {sorted(missing)}")
    steps = [item["step_id"] for item in case["scripted_observations"]]
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError(f"{case['case_id']}: step ids are not contiguous")
    if case["target_switch_step"] != steps[-1]:
        raise ValueError(f"{case['case_id']}: target switch must be final observation")
    text = "\n".join(item["content"] for item in case["scripted_observations"])
    expected = str(case["expected_answer"])
    if expected not in text:
        raise ValueError(f"{case['case_id']}: expected answer absent from evidence")
    lower, upper, _, _ = TIERS[case["length_tier"]]
    count = token_count(case, tokenizer)
    if not lower <= count <= upper:
        raise ValueError(f"{case['case_id']}: token count {count} outside tier")

    if text.count(expected) != 1:
        raise ValueError(f"{case['case_id']}: expected answer must occur exactly once")

def to_dataset_record(case: Dict[str, Any]) -> Dict[str, Any]:
    env_kwargs = {"case": case, "question": case["initial_goal"], "ground_truth": case["expected_answer"], "gt_traj": "", "data_source": "delayed_evidence"}
    return {
        "data_source": "delayed_evidence",
        "prompt": [{"role": "system", "content": "Follow the deterministic evidence protocol."}, {"role": "user", "content": case["initial_goal"]}],
        "ability": "context_recovery",
        "reward_model": {"style": "rule", "ground_truth": case["expected_answer"]},
        "extra_info": {"case_id": case["case_id"], "category": case["category"], "length_tier": case["length_tier"]},
        "metadata": json.dumps({"case_id": case["case_id"]}, sort_keys=True),
        "env_kwargs": env_kwargs,
    }


def build(output_dir: str, model: str, allow_fallback: bool = False, case_id: str = "") -> List[Dict[str, Any]]:
    tokenizer = load_tokenizer(model, allow_fallback)
    cases = [make_case(category, tier, variant, tokenizer) for tier in TIERS for category in CATEGORIES for variant in range(3)]
    if case_id:
        cases = [case for case in cases if case["case_id"] == case_id]
    for case in cases:
        validate_case(case, tokenizer)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases_path = out / "cases.jsonl"
    review_dir = out / "cases"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_files = []
    for case in cases:
        review_path = review_dir / f"{case["case_id"]}.json"
        review_path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        review_files.append({"file": str(review_path.relative_to(out)), "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest()})
    with open(cases_path, "w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    pd.DataFrame([to_dataset_record(case) for case in cases]).to_parquet(out / "test.parquet")
    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "case_count": len(cases),
        "categories": list(CATEGORIES),
        "tiers": {key: list(value[:2]) for key, value in TIERS.items()},
        "tokenizer": getattr(tokenizer, "name_or_path", model),
        "cases_sha256": digest,
        "review_files": review_files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/task_c")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--allow-fallback-tokenizer", action="store_true")
    parser.add_argument("--case-id", default="")
    args = parser.parse_args()
    cases = build(args.output_dir, args.model, args.allow_fallback_tokenizer, args.case_id)
    print(f"built {len(cases)} cases in {args.output_dir}")

if __name__ == "__main__":
    main()
