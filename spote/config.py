"""Paper settings and explicit choices for unspecified implementation details."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class SPOTEConfig:
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    clip_model: str = "openai/clip-vit-base-patch16"
    device: str = "cuda"
    precision: str = "bf16"
    steps: int = 8
    group_size: int = 4
    learning_rate: float = 5e-4
    lora_rank: int = 16
    lora_alpha: float = 16.
    lora_targets: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])
    anchors: int = 60
    preservation_weight: float = .5
    clarity_constant: float = 1.
    patch_features: str = "value_residual"
    mask_temperature: float = .7
    mask_calibration: str = "reference"
    noise_std: float = .01
    background_brightness: float = .2
    crop_fraction: float = .6
    clip_range: float = .2
    kl_weight: float = .01
    policy_epochs: int = 1
    max_new_tokens: int = 128
    max_pixels: int = 1024 * 28 * 28
    seed: int = 0

    def __post_init__(self):
        for key in ("group_size", "lora_rank", "anchors", "policy_epochs", "max_new_tokens", "max_pixels"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0 or self.group_size < 2:
            raise ValueError("steps must be nonnegative and group_size must be at least 2")
        for key in ("learning_rate", "lora_alpha", "clarity_constant", "mask_temperature", "noise_std"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("preservation_weight", "kl_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not 0 <= self.background_brightness <= 1 or not 0 < self.crop_fraction < 1 or not 0 < self.clip_range < 1:
            raise ValueError("invalid dimming, crop fraction or clipping range")
        if self.precision not in ("bf16", "fp32") or (self.device == "cpu" and self.precision != "fp32"):
            raise ValueError("use bf16/fp32, and fp32 on CPU")
        if self.mask_calibration not in ("reference", "none"):
            raise ValueError("mask_calibration must be reference or none")
        if self.patch_features not in ("value_residual", "last_hidden"):
            raise ValueError("patch_features must be value_residual or last_hidden")
        if (not self.lora_targets or len(set(self.lora_targets)) != len(self.lora_targets)
                or set(self.lora_targets) - {"q_proj", "k_proj", "v_proj", "out_proj"}):
            raise ValueError("LoRA targets must be unique CLIP attention projections")

    @classmethod
    def load(cls, path, **overrides):
        values = json.loads(Path(path).read_text())
        return cls(**{**values, **{k: v for k, v in overrides.items() if v is not None}})

    def to_dict(self):
        return asdict(self)
