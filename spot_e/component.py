"""High-level SPOT-E plug-in orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import SPOTEConfig
from .entropy import (
    anchor_disruption,
    entropy_shaping_reward,
    low_entropy_anchors,
)
from .grpo import build_grpo_batch
from .interfaces import CandidateEvaluation, FrozenVLMProbe, SpotlightPolicy, TokenTrace
from .spotlight import GridSpotlightPolicy, apply_spotlight


@dataclass(frozen=True)
class SPOTEResult:
    """Outputs returned by one SPOT-E test-time episode."""

    image: Image.Image
    mask: np.ndarray
    baseline_trace: TokenTrace
    selected_trace: TokenTrace
    selected_evaluation: CandidateEvaluation
    anchors: tuple[int, ...]
    candidate_evaluations: tuple[CandidateEvaluation, ...]


class SPOTEPlugin:
    """Plug-and-play SPOT-E component.

    The plugin keeps the user-provided VLM probe frozen, optimizes only the
    spotlight policy for the current instance, chooses the best candidate by
    entropy-shaping reward, and resets the policy after returning.
    """

    def __init__(
        self,
        probe: FrozenVLMProbe,
        *,
        policy: SpotlightPolicy | None = None,
        config: SPOTEConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.probe = probe
        self.config = config or SPOTEConfig()
        self.policy = policy or GridSpotlightPolicy(self.config.spotlight)
        self.rng = np.random.default_rng(seed)

    def run(self, image: Image.Image, question: str) -> SPOTEResult:
        """Run one per-instance SPOT-E episode and return the selected image."""

        self.policy.reset()
        baseline_trace = self.probe.trace(image, question)
        anchors = low_entropy_anchors(
            baseline_trace,
            self.config.reward.anchor_count,
        )

        all_masks: list[np.ndarray] = []
        all_evaluations: list[CandidateEvaluation] = []
        all_traces: list[TokenTrace] = []

        try:
            for _ in range(self.config.grpo.steps):
                masks = self.policy.sample(
                    image,
                    question,
                    group_size=self.config.grpo.group_size,
                    noise_std=self.config.spotlight.gaussian_noise_std,
                    rng=self.rng,
                )
                evaluations, traces = self._evaluate_masks(
                    image,
                    question,
                    masks,
                    baseline_trace,
                    anchors,
                )
                batch = build_grpo_batch(
                    [evaluation.reward for evaluation in evaluations],
                    self.config.grpo,
                )
                self.policy.update(
                    masks,
                    batch.advantages,
                    learning_rate=self.config.grpo.learning_rate,
                    clip_epsilon=self.config.grpo.clip_epsilon,
                    kl_beta=self.config.grpo.kl_beta,
                )
                all_masks.extend(masks)
                all_evaluations.extend(evaluations)
                all_traces.extend(traces)

            final_masks = [self.policy.score_mask(image, question)]
            if self.config.best_of_n > 1:
                final_masks.extend(
                    self.policy.sample(
                        image,
                        question,
                        group_size=self.config.best_of_n - 1,
                        noise_std=self.config.spotlight.gaussian_noise_std,
                        rng=self.rng,
                    )
                )
            final_evaluations, final_traces = self._evaluate_masks(
                image,
                question,
                final_masks,
                baseline_trace,
                anchors,
            )
            all_masks.extend(final_masks)
            all_evaluations.extend(final_evaluations)
            all_traces.extend(final_traces)

            best_index = int(np.argmax([evaluation.reward for evaluation in all_evaluations]))
            best_mask = all_masks[best_index]
            best_image = apply_spotlight(
                image,
                best_mask,
                background_strength=self.config.spotlight.background_strength,
            )
            return SPOTEResult(
                image=best_image,
                mask=best_mask,
                baseline_trace=baseline_trace,
                selected_trace=all_traces[best_index],
                selected_evaluation=all_evaluations[best_index],
                anchors=anchors,
                candidate_evaluations=tuple(all_evaluations),
            )
        finally:
            self.policy.reset()

    def _evaluate_masks(
        self,
        image: Image.Image,
        question: str,
        masks: list[np.ndarray],
        baseline_trace: TokenTrace,
        anchors: tuple[int, ...],
    ) -> tuple[list[CandidateEvaluation], list[TokenTrace]]:
        evaluations: list[CandidateEvaluation] = []
        traces: list[TokenTrace] = []
        for mask in masks:
            candidate_image = apply_spotlight(
                image,
                mask,
                background_strength=self.config.spotlight.background_strength,
            )
            candidate_trace = self.probe.trace(
                candidate_image,
                question,
                baseline_trace=baseline_trace,
            )
            reward, clarity, preserve, ans_entropy, delta_ans = entropy_shaping_reward(
                baseline_trace,
                candidate_trace,
                anchors,
                self.config.reward,
            )
            disruption = anchor_disruption(baseline_trace, candidate_trace, anchors)
            evaluations.append(
                CandidateEvaluation(
                    reward=reward,
                    clarity_reward=clarity,
                    preserve_reward=preserve,
                    answer_entropy=ans_entropy,
                    answer_entropy_delta=delta_ans,
                    anchor_disruption=disruption,
                )
            )
            traces.append(candidate_trace)
        return evaluations, traces
