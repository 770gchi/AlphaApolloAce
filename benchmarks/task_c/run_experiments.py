"""Run the preregistered Task-C AlphaApollo experiment matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RunSpec:
    name: str
    mode: str
    budget: int
    keep_last_raw: bool = True
    orchestration: str = "llm"
    allowed_types: str = "[raw,abstract,drop]"
    oracle: bool = False


def matrix(include_ablations: bool = True) -> List[RunSpec]:
    runs = [RunSpec(f"{mode}_llm_{budget // 1024}k" if mode == "elastic" else f"{mode}_{budget // 1024}k", mode, budget) for budget in (2048, 4096) for mode in ("full", "truncate", "summarize", "elastic")]
    if include_ablations:
        runs += [
            RunSpec("elastic_no_last_4k", "elastic", 4096, keep_last_raw=False),
            RunSpec("elastic_raw_drop_4k", "elastic", 4096, allowed_types="[raw,drop]"),
            RunSpec("elastic_heuristic_4k", "elastic", 4096, orchestration="heuristic"),
            RunSpec("full_20k_oracle", "full", 20000, oracle=True),
        ]
    return runs


def gpu_memory_mb() -> int:
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        return max(int(line.strip()) for line in output.splitlines() if line.strip())
    except Exception:
        return 0


def run_one(spec: RunSpec, root: Path, data_dir: Path, model_path: str, batch_size: int, dry_run: bool = False) -> None:
    output_scope = "task_c_trial" if data_dir.name.endswith("_trial") else "task_c"
    raw_dir = root / f"outputs/{output_scope}/raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    result_json = raw_dir / f"{spec.name}.jsonl"; result_parquet = raw_dir / f"{spec.name}.parquet"
    command = [
        sys.executable, "-m", "alphaapollo.core.generation.verl.trainer.main_generation",
        "trainer.nnodes=1", "trainer.n_gpus_per_node=1", f"data.path={data_dir / "test.parquet"}", "data.prompt_key=prompt",
        "data.n_samples=1", f"data.batch_size={batch_size}", "+data.dataloader_num_workers=0", "data.return_raw_chat=true", "data.truncation=right",
        f"data.output_path={result_parquet}", "data.save2json=true", f"data.json_output_path={result_json}",
        f"model.path={model_path}", "rollout.name=vllm", "rollout.do_sample=false", "rollout.temperature=0.0",
        "rollout.top_k=-1", "rollout.top_p=1.0", f"rollout.prompt_length={spec.budget}", "rollout.response_length=512",
        "rollout.max_model_len=32768", "rollout.tensor_model_parallel_size=1", "rollout.gpu_memory_utilization=0.75",
        f"rollout.max_num_batched_tokens={max(8192, spec.budget)}", f"context_policy.mode={spec.mode}",
        "rollout.enable_chunked_prefill=false",
        f"context_policy.max_input_tokens={spec.budget}", "context_policy.reserve_output_tokens=512", "context_policy.model_context_window=32768",
        f"context_policy.keep_last_raw={str(spec.keep_last_raw).lower()}", "context_policy.abstraction_cache=true",
        f"context_policy.orchestration={spec.orchestration}", "context_policy.truncate_window_steps=4",
        "context_policy.abstraction_max_tokens=256", "context_policy.orchestration_max_tokens=512",
        f"context_policy.allowed_types={spec.allowed_types}", f"context_policy.audit_dir=./outputs/{output_scope}/audit/{spec.name}",
        f"context_policy.store_dir=./outputs/{output_scope}/store/{spec.name}", "env.env_name=delayed_evidence", "env.seed=0",
        "env.max_steps=17", "env.history_length=16", "env.resources_per_worker.num_cpus=0.1",
    ]
    if dry_run:
        print(" ".join(command)); return
    started = time.time(); peak = gpu_memory_mb(); proc = subprocess.Popen(command, cwd=root)
    while proc.poll() is None:
        peak = max(peak, gpu_memory_mb()); time.sleep(0.5)
    elapsed = time.time() - started
    metadata = {**asdict(spec), "command": command, "exit_code": proc.returncode, "wall_seconds": elapsed, "peak_gpu_memory_mb": peak}
    meta_path = root / f"outputs/{output_scope}/manifests/{spec.name}.json"; meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"run {spec.name} failed with exit code {proc.returncode}")
    subprocess.run([
        sys.executable, "-m", "benchmarks.task_c.evaluate", "--cases", str(data_dir / "cases.jsonl"), "--output", str(result_json),
        "--policy", spec.name, "--budget", str(spec.budget), "--output-dir", f"outputs/{output_scope}/metrics/{spec.name}",
    ], cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--case-id", default=""); parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--model-path", default=os.environ.get("TASK_C_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--batch-size", type=int, default=30); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-fallback-tokenizer", action="store_true")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[2]
    data_dir = Path("data/task_c_trial" if args.case_id else "data/task_c")
    build = [sys.executable, "-m", "benchmarks.task_c.build", "--output-dir", str(data_dir), "--model", args.model_path]
    if args.case_id: build += ["--case-id", args.case_id]
    if args.allow_fallback_tokenizer: build.append("--allow-fallback-tokenizer")
    if args.dry_run: print(" ".join(build))
    else: subprocess.run(build, cwd=root, check=True)
    runs = matrix(not args.core_only)
    if args.run:
        wanted = set(args.run); known = {spec.name for spec in runs}
        unknown = wanted - known
        if unknown: raise SystemExit(f"unknown run(s): {", ".join(sorted(unknown))}; choose from {", ".join(sorted(known))}")
        runs = [spec for spec in runs if spec.name in wanted]
    for spec in runs:
        if args.resume:
            meta_path = root / f"outputs/task_c/manifests/{spec.name}.json"
            raw_path = root / f"outputs/task_c/raw/{spec.name}.jsonl"
            metric_path = root / f"outputs/task_c/metrics/{spec.name}/summary.csv"
            if meta_path.exists() and raw_path.exists() and metric_path.exists() and json.loads(meta_path.read_text()).get("exit_code") == 0:
                print(f"skip completed run: {spec.name}"); continue
        run_one(spec, root, data_dir, args.model_path, args.batch_size, args.dry_run)


if __name__ == "__main__": main()
