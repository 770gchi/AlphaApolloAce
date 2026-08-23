"""Generate Task-C publication/slide figures from saved CSV/JSONL outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICY_LABELS = {
    "full": "Full",
    "truncate": "Truncate",
    "summarize": "Summarize",
    "elastic": "Elastic-LLM",
}
COLORS = {
    "Full": "#7f8c8d",
    "Truncate": "#e67e22",
    "Summarize": "#3498db",
    "Elastic-LLM": "#2ecc71",
}


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def read_case(metrics: Path, run: str) -> pd.DataFrame:
    return pd.read_csv(metrics / run / "per_case.csv")


def accuracy_plot(metrics: Path, out: Path) -> None:
    policies = ["full", "truncate", "summarize", "elastic"]
    budgets = [2048, 4096]
    values = {}
    for budget in budgets:
        suffix = f"{budget // 1024}k"
        for policy in policies:
            run = f"elastic_llm_{suffix}" if policy == "elastic" else f"{policy}_{suffix}"
            values[(policy, budget)] = read_case(metrics, run)["correct"].mean() * 100
    x = np.arange(len(policies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for idx, budget in enumerate(budgets):
        vals = [values[(p, budget)] for p in policies]
        bars = ax.bar(x + (idx - 0.5) * width, vals, width, label=f"{budget // 1024}K budget",
                      color=["#95a5a6", "#f39c12", "#5dade2", "#58d68d"],
                      alpha=0.72 + idx * 0.2, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, [POLICY_LABELS[p] for p in policies])
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Task C2: Accuracy under context pressure")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out / "c2_accuracy.png")


def ablation_plot(metrics: Path, out: Path) -> None:
    runs = [
        ("elastic_llm_4k", "Elastic-LLM"),
        ("elastic_no_last_4k", "No latest RAW"),
        ("elastic_raw_drop_4k", "RAW/DROP only"),
        ("elastic_heuristic_4k", "Heuristic"),
    ]
    vals = [read_case(metrics, run)["correct"].mean() * 100 for run, _ in runs]
    labels = [label for _, label in runs]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(labels, vals, color=["#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"],
                  edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Task C3: Elastic ablations at 4K")
    ax.grid(axis="y", alpha=0.25)
    save(fig, out / "c3_ablations.png")


def token_cost_plot(metrics: Path, out: Path) -> None:
    runs = [
        ("full_4k", "Full"),
        ("truncate_4k", "Truncate"),
        ("summarize_4k", "Summarize"),
        ("elastic_llm_4k", "Elastic-LLM"),
    ]
    prompt = []
    management = []
    for run, _ in runs:
        frame = read_case(metrics, run)
        prompt.append(frame["mean_prompt_tokens"].mean())
        management.append(frame["management_tokens"].sum() / len(frame))
    labels = [label for _, label in runs]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    bars = axes[0].bar(labels, prompt, color=[COLORS[x] for x in labels])
    axes[0].bar_label(bars, fmt="%.0f", padding=3)
    axes[0].set_title("Mean main-prompt tokens / turn")
    axes[0].set_ylabel("Tokens")
    axes[0].grid(axis="y", alpha=0.25)
    bars = axes[1].bar(labels, management, color=[COLORS[x] for x in labels])
    axes[1].bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    axes[1].set_title("Management tokens / trajectory")
    axes[1].set_ylabel("Tokens")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Task C2: Context and management cost at 4K")
    save(fig, out / "token_costs_4k.png")


def type_over_turn_plot(metrics: Path, out: Path) -> None:
    frame = pd.read_json(metrics / "elastic_llm_4k" / "per_step.jsonl", lines=True)
    counts = frame.groupby(["decision_step", "elastic_type"]).size().unstack(fill_value=0)
    for kind in ["raw", "abstract", "drop"]:
        if kind not in counts:
            counts[kind] = 0
    proportions = counts[["drop", "abstract", "raw"]].div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.stackplot(
        proportions.index,
        proportions["drop"],
        proportions["abstract"],
        proportions["raw"],
        labels=["DROP", "ABSTRACT", "RAW"],
        colors=["#7f8c8d", "#3498db", "#2ecc71"],
        alpha=0.9,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Decision turn")
    ax.set_ylabel("Fraction of historical steps")
    ax.set_title("Elastic-LLM 4K: Representation mix over turns")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, out / "elastic_types_over_turn.png")


def full_telemetry_plot(raw_dir: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (run, budget) in zip(axes, [("full_2k", 2048), ("full_4k", 4096)]):
        rows = []
        with open(raw_dir / f"{run}.jsonl", encoding="utf-8") as stream:
            for line in stream:
                sample = json.loads(line)
                for records in sample.get("trajectory_records", []):
                    for record in records:
                        event = record.get("truncation_telemetry") or {}
                        rows.append({
                            "step": record.get("step_id", 0),
                            "pre": event.get("pre_truncation_tokens", 0),
                            "post": event.get("post_truncation_tokens", 0),
                        })
        frame = pd.DataFrame(rows)
        grouped = frame.groupby("step")[["pre", "post"]].mean()
        ax.plot(grouped.index, grouped["pre"], marker="o", label="Before truncation", color="#e74c3c")
        ax.plot(grouped.index, grouped["post"], marker="o", label="Model input", color="#3498db")
        ax.axhline(budget, color="black", linestyle="--", linewidth=1, label=f"{budget // 1024}K budget")
        ax.fill_between(grouped.index, grouped["post"], grouped["pre"], color="#e74c3c", alpha=0.12)
        ax.set_title(f"Full-{budget // 1024}K")
        ax.set_xlabel("Rollout step")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean prompt tokens")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Full baseline: Actual right-truncation telemetry", y=1.04)
    save(fig, out / "full_truncation_telemetry.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="outputs/task_c/metrics")
    parser.add_argument("--raw-dir", default="outputs/task_c/raw")
    parser.add_argument("--output-dir", default="outputs/task_c/plots")
    args = parser.parse_args()
    metrics = Path(args.metrics_dir)
    raw = Path(args.raw_dir)
    out = Path(args.output_dir)
    accuracy_plot(metrics, out)
    ablation_plot(metrics, out)
    token_cost_plot(metrics, out)
    type_over_turn_plot(metrics, out)
    full_telemetry_plot(raw, out)


if __name__ == "__main__":
    main()
