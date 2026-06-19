"""Entropy, anchors, and reward functions from SPOT-E."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .config import RewardConfig
from .interfaces import TokenTrace, ensure_trace_lengths


def shannon_entropy_from_probs(probs: Sequence[float], *, eps: float = 1e-12) -> float:
    """Compute Shannon entropy with natural logarithms."""

    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("probs must be a one-dimensional distribution.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("probs must contain positive mass.")
    arr = np.clip(arr / total, eps, 1.0)
    return float(-np.sum(arr * np.log(arr)))


def shannon_entropy_from_logits(logits: Sequence[float]) -> float:
    """Compute Shannon entropy from unnormalized logits."""

    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    probs /= probs.sum()
    return shannon_entropy_from_probs(probs)


def answer_entropy(trace: TokenTrace) -> float:
    """Average next-token entropy over the final answer span."""

    ensure_trace_lengths(trace)
    values = [float(trace.entropies[i]) for i in trace.answer_indices()]
    if not values:
        raise ValueError("answer_span must include at least one token.")
    return float(np.mean(values))


def low_entropy_anchors(trace: TokenTrace, k: int) -> tuple[int, ...]:
    """Select the K positions with smallest baseline next-token entropy."""

    ensure_trace_lengths(trace)
    if k <= 0:
        return ()
    entropies = np.asarray(trace.entropies, dtype=np.float64)
    count = min(int(k), len(entropies))
    indices = np.argsort(entropies, kind="stable")[:count]
    return tuple(int(i) for i in indices)


def anchor_disruption(
    baseline_trace: TokenTrace,
    candidate_trace: TokenTrace,
    anchors: Iterable[int],
) -> float:
    """Average positive entropy increase on low-entropy anchor positions."""

    ensure_trace_lengths(baseline_trace)
    ensure_trace_lengths(candidate_trace)
    values: list[float] = []
    for index in anchors:
        if index >= len(candidate_trace.entropies):
            continue
        delta = float(candidate_trace.entropies[index]) - float(
            baseline_trace.entropies[index]
        )
        values.append(max(0.0, delta))
    if not values:
        return 0.0
    return float(np.mean(values))


def dynamic_clarity_scale(baseline_answer_entropy: float, c: float) -> float:
    """Paper Eq. 16: gamma = H_ans(x,q) / (H_ans(x,q) + c)."""

    if c <= 0:
        raise ValueError("dynamic_scale_c must be positive.")
    h = max(0.0, float(baseline_answer_entropy))
    return h / (h + c)


def entropy_shaping_reward(
    baseline_trace: TokenTrace,
    candidate_trace: TokenTrace,
    anchors: Sequence[int],
    config: RewardConfig,
) -> tuple[float, float, float, float, float]:
    """Return total reward and its paper-defined components.

    The answer entropy delta follows paper Eq. 15:
    H_ans(baseline) - H_ans(candidate), so positive values mean the spotlight
    made the final answer span more decisive.
    """

    baseline_ans = answer_entropy(baseline_trace)
    candidate_ans = answer_entropy(candidate_trace)
    delta_ans = baseline_ans - candidate_ans
    gamma = dynamic_clarity_scale(baseline_ans, config.dynamic_scale_c)
    clarity = gamma * delta_ans
    disruption = anchor_disruption(baseline_trace, candidate_trace, anchors)
    preserve = -float(config.preserve_lambda) * disruption
    reward = clarity + preserve
    return float(reward), float(clarity), float(preserve), float(candidate_ans), float(delta_ans)


def group_relative_advantages(
    rewards: Sequence[float],
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Paper Eq. 18-19 standardized group-relative advantages."""

    arr = np.asarray(rewards, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("rewards must be a non-empty one-dimensional sequence.")
    return (arr - arr.mean()) / (arr.std() + eps)
