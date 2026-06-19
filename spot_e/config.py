"""Configuration objects for SPOT-E."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """Hyperparameters for the entropy-shaping reward."""

    anchor_count: int = 8
    dynamic_scale_c: float = 0.1
    preserve_lambda: float = 0.5
    eps: float = 1e-8


@dataclass(frozen=True)
class SpotlightConfig:
    """Hyperparameters for mask construction and image intervention."""

    grid_size: tuple[int, int] = (14, 14)
    temperature: float = 0.25
    background_strength: float = 0.35
    gaussian_noise_std: float = 0.15
    min_mask_value: float = 0.0
    max_mask_value: float = 1.0


@dataclass(frozen=True)
class GRPOConfig:
    """Small per-instance GRPO episode settings."""

    steps: int = 2
    group_size: int = 4
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    learning_rate: float = 0.05
    eps: float = 1e-8


@dataclass(frozen=True)
class SPOTEConfig:
    """Top-level SPOT-E configuration."""

    reward: RewardConfig = field(default_factory=RewardConfig)
    spotlight: SpotlightConfig = field(default_factory=SpotlightConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    best_of_n: int = 4
