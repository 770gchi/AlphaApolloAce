from typing import List, Tuple


def delayed_evidence_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """The scripted environment preserves the complete model action."""
    return [str(action or "") for action in actions], [1 for _ in actions]
