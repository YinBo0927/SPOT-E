"""Group-relative optimization helpers for SPOT-E."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import GRPOConfig
from .entropy import group_relative_advantages


@dataclass(frozen=True)
class GRPOBatch:
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]


def build_grpo_batch(rewards: Sequence[float], config: GRPOConfig) -> GRPOBatch:
    """Compute group-relative advantages for one sampled candidate group."""

    advantages = group_relative_advantages(rewards, eps=config.eps)
    return GRPOBatch(
        rewards=tuple(float(value) for value in rewards),
        advantages=tuple(float(value) for value in advantages),
    )
