"""SPOT-E plug-and-play test-time visual spotlight component."""

from .component import SPOTEPlugin, SPOTEResult
from .config import (
    GRPOConfig,
    RewardConfig,
    SPOTEConfig,
    SpotlightConfig,
)
from .entropy import (
    answer_entropy,
    anchor_disruption,
    entropy_shaping_reward,
    low_entropy_anchors,
)
from .interfaces import (
    CandidateEvaluation,
    FrozenVLMProbe,
    SpotlightPolicy,
    TokenTrace,
)
from .spotlight import (
    CropRelevance,
    GridSpotlightPolicy,
    apply_spotlight,
    background_degrade,
    compact_visual_phrase,
    fuse_relevance_maps,
    mask_to_image,
    relevance_to_mask,
)

__all__ = [
    "CandidateEvaluation",
    "CropRelevance",
    "FrozenVLMProbe",
    "GRPOConfig",
    "GridSpotlightPolicy",
    "RewardConfig",
    "SPOTEConfig",
    "SPOTEPlugin",
    "SPOTEResult",
    "SpotlightConfig",
    "SpotlightPolicy",
    "TokenTrace",
    "anchor_disruption",
    "answer_entropy",
    "apply_spotlight",
    "background_degrade",
    "compact_visual_phrase",
    "entropy_shaping_reward",
    "fuse_relevance_maps",
    "low_entropy_anchors",
    "mask_to_image",
    "relevance_to_mask",
]
