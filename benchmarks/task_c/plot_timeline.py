from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--per-step", required=True); parser.add_argument("--case-id", required=True); parser.add_argument("--output", required=True); parser.add_argument("--title", default="")
    args = parser.parse_args(); import matplotlib.pyplot as plt
    df = pd.read_json(args.per_step, lines=True); df = df[df.case_id == args.case_id]
    if df.empty:
        raise ValueError(f"no timeline records for case_id {args.case_id!r}")
    pivot = df.pivot_table(index="history_step", columns="decision_step", values="elastic_type", aggfunc="last")
    values = pivot.replace({"drop": 0, "abstract": 1, "raw": 2}).fillna(-1).astype(float)
    fig, ax = plt.subplots(figsize=(10, 6)); image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=2)
    ax.set_xlabel("Decision turn"); ax.set_ylabel("Historical step"); ax.set_title(args.title or args.case_id)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns); ax.set_yticks(range(len(pivot.index)), pivot.index)
    colorbar = fig.colorbar(image, ticks=[0, 1, 2]); colorbar.ax.set_yticklabels(["DROP", "ABSTRACT", "RAW"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(args.output, dpi=180); plt.close(fig)


if __name__ == "__main__": main()
