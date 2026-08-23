"""Offline ACE replay and audit integrity checker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .models import Decision, ElasticType
from .store import LosslessStore


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def replay(store_path: str, audit_path: str, output_path: str, recompute_policy: bool = False) -> int:
    store = LosslessStore(store_path)
    audits = _read_jsonl(audit_path)
    output: List[Dict[str, Any]] = []
    errors: List[str] = []
    previous_step: Dict[str, int] = {}
    for audit in audits:
        tid = str(audit["trajectory_id"])
        decision_step = int(audit["decision_step"])
        if decision_step < previous_step.get(tid, 0):
            errors.append(f"{tid}:{decision_step}: non-monotonic decision step")
        previous_step[tid] = decision_step
        for sid_text, expected_hash in audit.get("source_hashes", {}).items():
            rec = store.get(int(sid_text), tid)
            if rec is None or rec.source_hash != expected_hash:
                errors.append(f"{tid}:{decision_step}: source hash mismatch at step {sid_text}")
        prompt = audit.get("final_prompt", "")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest != audit.get("final_message_hash"):
            errors.append(f"{tid}:{decision_step}: final message hash mismatch")
        decisions = audit.get("decisions", [])
        if recompute_policy:
            # Offline deterministic policy recomputation uses recorded raw truth;
            # it intentionally performs no model or environment call.
            task_words = set(str(store.get(1, tid).metadata.get("task", "")).lower().split()) if store.get(1, tid) else set()
            recomputed = []
            steps = [store.get(sid, tid) for sid in store.steps(tid) if sid <= decision_step]
            for rec in steps:
                overlap = len(task_words & set(" ".join(rec.raw_messages).lower().split()))
                kind = ElasticType.RAW.value if rec.step_id == decision_step or overlap else ElasticType.DROP.value
                recomputed.append(Decision(rec.step_id, ElasticType(kind), "offline heuristic", float(bool(overlap))).to_dict())
            decisions = recomputed
        output.append({
            "trajectory_id": tid,
            "decision_step": decision_step,
            "observation": audit.get("final_observation", ""),
            "prompt": prompt,
            "decisions": decisions,
            "message_hash": digest,
        })
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    for error in errors:
        print(error)
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild and verify ACE context views without an environment")
    parser.add_argument("--store", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--config", help="Accepted for CLI compatibility; recorded audit is authoritative")
    parser.add_argument("--output", required=True)
    parser.add_argument("--recompute-policy", action="store_true")
    args = parser.parse_args()
    raise SystemExit(replay(args.store, args.audit, args.output, args.recompute_policy))


if __name__ == "__main__":
    main()
