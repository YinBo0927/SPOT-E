"""Question-conditioned visual spotlight policy and image intervention."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import SpotlightConfig


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "color",
    "does",
    "for",
    "in",
    "is",
    "of",
    "on",
    "the",
    "there",
    "to",
    "what",
    "which",
    "with",
}


def compact_visual_phrase(question: str) -> str:
    """Tiny default phrase extractor for q-bar.

    Real deployments should replace this with an NLP parser or an LLM prompt if
    they want the same key-entity behavior described in the paper.
    """

    words = re.findall(r"[A-Za-z0-9]+", question.lower())
    kept = [word for word in words if word not in _STOPWORDS]
    return " ".join(kept[:12]) or question.strip()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_map(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


@dataclass(frozen=True)
class CropRelevance:
    """A crop-level relevance map and its box in full-map coordinates.

    `box` is `(left, top, right, bottom)` in the target relevance-map grid, not
    image pixels. For example, on a 14x14 map, `(0, 0, 7, 7)` denotes the
    upper-left quadrant.
    """

    relevance: np.ndarray
    box: tuple[int, int, int, int]


def warp_crop_relevance(
    crop: CropRelevance,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp one crop relevance map back to full-image map coordinates."""

    out_h, out_w = output_shape
    left, top, right, bottom = crop.box
    left = int(np.clip(left, 0, out_w))
    right = int(np.clip(right, 0, out_w))
    top = int(np.clip(top, 0, out_h))
    bottom = int(np.clip(bottom, 0, out_h))
    canvas = np.full((out_h, out_w), -np.inf, dtype=np.float32)
    if right <= left or bottom <= top:
        return canvas

    crop_map = np.asarray(crop.relevance, dtype=np.float32)
    image = Image.fromarray(np.uint8(normalize_map(crop_map) * 255), mode="L")
    image = image.resize((right - left, bottom - top), Image.Resampling.BILINEAR)
    warped = np.asarray(image, dtype=np.float32) / 255.0
    canvas[top:bottom, left:right] = warped
    return canvas


def fuse_relevance_maps(
    global_relevance: np.ndarray,
    crops: Sequence[CropRelevance] = (),
) -> np.ndarray:
    """Paper Eq. 11 max-fusion over global and crop relevance maps."""

    global_map = normalize_map(global_relevance)
    fused = global_map.astype(np.float32)
    for crop in crops:
        warped = warp_crop_relevance(crop, fused.shape)
        fused = np.maximum(fused, warped)
    return normalize_map(fused)


def relevance_to_mask(relevance: np.ndarray, *, temperature: float) -> np.ndarray:
    """Paper Eq. 12 soft mask from an upsample-ready relevance map."""

    centered = normalize_map(relevance) * 2.0 - 1.0
    return sigmoid(centered / max(float(temperature), 1e-6)).astype(np.float32)


def upsample_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a grid mask to image size as a float array in [0, 1]."""

    arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), mode="L")
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def background_degrade(
    image: Image.Image,
    *,
    strength: float = 0.35,
    blur_radius: float = 1.2,
) -> Image.Image:
    """Fixed background-degrading transform B(x).

    The paper uses background dimming; this implementation dims and lightly
    blurs the background to make foreground evidence easier to read.
    """

    strength = float(np.clip(strength, 0.0, 1.0))
    base = image.convert("RGB")
    dimmed = ImageEnhance.Brightness(base).enhance(1.0 - strength)
    if blur_radius > 0:
        dimmed = dimmed.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return dimmed


def apply_spotlight(
    image: Image.Image,
    mask: np.ndarray,
    *,
    background_strength: float = 0.35,
) -> Image.Image:
    """Paper Eq. 13: x_tilde = m*x + (1-m)*B(x)."""

    base = image.convert("RGB")
    full_mask = upsample_mask(mask, base.size)
    background = background_degrade(base, strength=background_strength)
    x = np.asarray(base, dtype=np.float32)
    b = np.asarray(background, dtype=np.float32)
    m = full_mask[:, :, None]
    out = m * x + (1.0 - m) * b
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)), mode="RGB")


def mask_to_image(mask: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    """Convert a mask to a grayscale PIL image for debugging or visualization."""

    arr = normalize_map(mask)
    image = Image.fromarray(np.uint8(arr * 255), mode="L")
    if size is not None:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


@dataclass
class GridSpotlightPolicy:
    """A lightweight policy that can be swapped for CLIP+LoRA.

    The paper computes CLIP patch-text similarities from global and crop views,
    then fuses them by max pooling. This default policy provides the same mask
    policy contract without shipping a CLIP model: callers may seed it with any
    external relevance map through `set_relevance_map`, and GRPO-like updates
    adjust the grid logits per instance.
    """

    config: SpotlightConfig = field(default_factory=SpotlightConfig)
    _initial_logits: np.ndarray = field(init=False, repr=False)
    _logits: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._initial_logits = np.zeros(self.config.grid_size, dtype=np.float32)
        self._logits = self._initial_logits.copy()

    def reset(self) -> None:
        self._logits = self._initial_logits.copy()

    def set_relevance_map(self, relevance: np.ndarray) -> None:
        """Initialize policy logits from a CLIP-style fused relevance map."""

        arr = np.asarray(relevance, dtype=np.float32)
        if arr.shape != self.config.grid_size:
            image = Image.fromarray(np.uint8(normalize_map(arr) * 255), mode="L")
            width, height = self.config.grid_size[1], self.config.grid_size[0]
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = normalize_map(arr)
        self._initial_logits = arr * 2.0 - 1.0
        self._logits = self._initial_logits.copy()

    def score_mask(self, image: Image.Image, question: str) -> np.ndarray:
        del image, question
        temperature = max(float(self.config.temperature), 1e-6)
        mask = sigmoid(self._logits / temperature)
        return np.clip(
            mask,
            self.config.min_mask_value,
            self.config.max_mask_value,
        ).astype(np.float32)

    def sample(
        self,
        image: Image.Image,
        question: str,
        *,
        group_size: int,
        noise_std: float,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        del image, question
        masks: list[np.ndarray] = []
        for _ in range(group_size):
            noise = rng.normal(0.0, noise_std, size=self._logits.shape)
            logits = self._logits + noise
            masks.append(sigmoid(logits / max(self.config.temperature, 1e-6)).astype(np.float32))
        return masks

    def update(
        self,
        masks: Sequence[np.ndarray],
        advantages: Sequence[float],
        *,
        learning_rate: float,
        clip_epsilon: float,
        kl_beta: float,
    ) -> None:
        """Small policy update inspired by clipped GRPO.

        This is a framework implementation, not a replacement for autograd over
        CLIP+LoRA. It moves logits toward above-average masks and adds a KL-like
        pull back to the per-instance initialization.
        """

        if not masks:
            return
        adv = np.asarray(advantages, dtype=np.float32)
        clipped_adv = np.clip(adv, -1.0 - clip_epsilon, 1.0 + clip_epsilon)
        stacked = np.stack([np.asarray(mask, dtype=np.float32) for mask in masks], axis=0)
        centered = stacked - sigmoid(self._logits / max(self.config.temperature, 1e-6))
        grad = np.mean(centered * clipped_adv[:, None, None], axis=0)
        kl_pull = self._logits - self._initial_logits
        self._logits = self._logits + learning_rate * grad - kl_beta * kl_pull
