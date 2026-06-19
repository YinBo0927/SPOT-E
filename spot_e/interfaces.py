"""Protocol interfaces used to plug SPOT-E into a real VLM stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TokenTrace:
    """Token-level readout from a frozen VLM.

    Entropies must align with the generated token sequence. The answer span is a
    half-open interval over token positions, usually extracted from a structured
    response such as "Final answer: ...".
    """

    tokens: Sequence[str]
    entropies: Sequence[float]
    answer_span: tuple[int, int]
    text: str = ""
    metadata: dict[str, Any] | None = None

    def answer_indices(self) -> range:
        start, end = self.answer_span
        return range(start, end)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Reward decomposition for one spotlight candidate."""

    reward: float
    clarity_reward: float
    preserve_reward: float
    answer_entropy: float
    answer_entropy_delta: float
    anchor_disruption: float


class FrozenVLMProbe(Protocol):
    """Minimal model adapter SPOT-E needs from a frozen VLM.

    Implementations should keep the underlying model frozen and return
    next-token entropies for the generated trajectory. For anchor comparison,
    `baseline_trace` is provided so adapters can condition on the baseline
    prefix and align positions, as described in the paper.
    """

    def trace(
        self,
        image: Image.Image,
        question: str,
        *,
        baseline_trace: TokenTrace | None = None,
    ) -> TokenTrace:
        ...


class SpotlightPolicy(Protocol):
    """Question-conditioned mask policy updated during one test-time episode."""

    def reset(self) -> None:
        ...

    def sample(
        self,
        image: Image.Image,
        question: str,
        *,
        group_size: int,
        noise_std: float,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        ...

    def update(
        self,
        masks: Sequence[np.ndarray],
        advantages: Sequence[float],
        *,
        learning_rate: float,
        clip_epsilon: float,
        kl_beta: float,
    ) -> None:
        ...

    def score_mask(self, image: Image.Image, question: str) -> np.ndarray:
        ...


def ensure_trace_lengths(trace: TokenTrace) -> None:
    """Raise early when a model adapter returns inconsistent token metadata."""

    if len(trace.tokens) != len(trace.entropies):
        raise ValueError("TokenTrace.tokens and TokenTrace.entropies must align.")
    start, end = trace.answer_span
    if start < 0 or end < start or end > len(trace.entropies):
        raise ValueError("TokenTrace.answer_span must be a valid half-open range.")
