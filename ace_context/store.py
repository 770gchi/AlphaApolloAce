"""A1: Lossless message maintenance layer.

The store is append-only. Raw messages are never overwritten or replaced by
abstractions; abstractions are cached and automatically invalidated when the
source content changes (source_hash mismatch). Supports JSONL persistence and
deterministic replay.
"""
from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

from .models import Abstraction, StepRecord

PathLike = Union[str, Path]


class LosslessStore:
    def __init__(self, path: Optional[PathLike] = None):
        # 以 trajectory 为第一层索引；不同 rollout 的 step_id 通常都会从 1 开始。
        self._steps: Dict[str, Dict[int, StepRecord]] = {}
        self.path: Optional[Path] = Path(path) if path else None
        if self.path and self.path.exists():
            self.load(self.path)

    def _trajectory(self, trajectory_id: Optional[str]) -> str:
        """兼容单 trajectory 调用；多 trajectory 时必须显式指定，防止串轨迹。"""
        if trajectory_id is not None:
            return trajectory_id
        if len(self._steps) == 1:
            return next(iter(self._steps))
        raise ValueError("trajectory_id is required when multiple trajectories exist")

    # ---------- write side ----------
    def append(self, record: StepRecord) -> None:
        bucket = self._steps.setdefault(record.trajectory_id, {})
        if record.step_id in bucket:
            raise ValueError(f"step {record.step_id} already exists in {record.trajectory_id}")
        # 复制调用方传入的 list，避免调用方随后修改 raw truth。
        bucket[record.step_id] = deepcopy(record)

    # ---------- read side ----------
    def get(self, step_id: int, trajectory_id: Optional[str] = None) -> Optional[StepRecord]:
        rec = self._steps.get(self._trajectory(trajectory_id), {}).get(step_id)
        return deepcopy(rec) if rec is not None else None

    def get_raw(self, step_id: int, trajectory_id: Optional[str] = None) -> Optional[List[str]]:
        rec = self._steps.get(self._trajectory(trajectory_id), {}).get(step_id)
        return rec.get_raw() if rec else None

    def get_abstraction(self, step_id: int, trajectory_id: Optional[str] = None) -> Optional[Abstraction]:
        """Return cached abstraction only if its source is still current."""
        rec = self._steps.get(self._trajectory(trajectory_id), {}).get(step_id)
        if rec is None or rec.abstraction is None:
            return None
        if rec.abstraction.source_hash != rec.source_hash:
            return None  # stale cache -> treat as missing
        return deepcopy(rec.abstraction)

    def put_abstraction(self, step_id: int, trajectory_id: Union[str, Abstraction], abstraction: Optional[Abstraction] = None) -> None:
        # 保留旧版单 trajectory API：put_abstraction(step_id, abstraction)。
        if isinstance(trajectory_id, Abstraction):
            abstraction = trajectory_id
            trajectory_id = self._trajectory(None)
        assert abstraction is not None
        rec = self._steps.get(trajectory_id, {}).get(step_id)
        if rec is None:
            raise KeyError(f"no step {step_id}")
        abstraction.source_step = step_id
        abstraction.source_hash = rec.source_hash
        rec.abstraction = abstraction

    def steps(self, trajectory_id: Optional[str] = None) -> List[int]:
        return sorted(self._steps.get(self._trajectory(trajectory_id), {}).keys())

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._steps.values())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, tuple) and len(key) == 2 and key[1] in self._steps.get(key[0], {})

    # ---------- persistence (JSONL) ----------
    def save(self, path: Optional[PathLike] = None) -> str:
        out = Path(path) if path else self.path
        if out is None:
            raise ValueError("no path given")
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", dir=str(out.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for trajectory_id in sorted(self._steps):
                    for sid in self.steps(trajectory_id):
                        f.write(json.dumps(self._steps[trajectory_id][sid].to_dict(), ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, out)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        return str(out)

    def trajectories(self) -> List[str]:
        return sorted(self._steps)

    def load(self, path: PathLike) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = StepRecord.from_dict(json.loads(line))
                self.append(rec)

    # ---------- deterministic replay ----------
    def replay(self, decisions: List[object]) -> None:
        """Not needed for the store itself; provided as a no-op placeholder.

        Replay of context views happens at the policy layer: given a recorded
        trajectory, each decision point rebuilds the same view without calling
        the environment. See policy.replay_view().
        """
        return None
