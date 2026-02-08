import hashlib
import math
from collections.abc import Sequence

from foundry.datasets.schemas import LabeledTweet


def _rank(seed: int, item_id: str) -> str:
    return hashlib.sha256(f'{seed}:{item_id}'.encode()).hexdigest()


def pick_audit_sample(
    batch: Sequence[LabeledTweet], rate: float, seed: int
) -> list[LabeledTweet]:
    """Select ceil(rate * n) rows for human review. Independent of input order."""
    if not 0 < rate <= 1:
        raise ValueError('rate must be in (0, 1]')
    if not batch:
        return []
    count = min(len(batch), math.ceil(rate * len(batch)))
    ranked = sorted(batch, key=lambda row: _rank(seed, row.id))
    return ranked[:count]
