"""CLIP visual LoRA, global/crop max fusion and background dimming."""
import math

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class SpotlightPolicy(nn.Module):
    def __init__(self, clip, processor, config):
        super().__init__()
        self.clip, self.processor, self.config = clip, processor, config
        clip.eval().requires_grad_(False)
        injected = 0
        for module in clip.vision_model.modules():
            for name in config.lora_targets:
                layer = getattr(module, name, None)
                if isinstance(layer, nn.Linear):
                    setattr(module, name, LoRALinear(layer, config.lora_rank, config.lora_alpha))
                    injected += 1
        if not injected:
            raise ValueError("no CLIP visual attention projections found")
        self.initial = {name: p.detach().clone() for name, p in self.named_parameters() if p.requires_grad}

    @classmethod
    def from_pretrained(cls, config):
        from transformers import CLIPModel, CLIPProcessor
        # LoRA initialization is repeatable without changing the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model = CLIPModel.from_pretrained(config.clip_model)
            policy = cls(model, CLIPProcessor.from_pretrained(config.clip_model), config)
        return policy.to(config.device).eval()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def reset(self):
        for name, p in self.named_parameters():
            if name in self.initial:
                p.copy_(self.initial[name].to(p))
                p.grad = None

    def prepare(self, image, phrase):
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("visual phrase must be nonempty")
        w, h = image.size
        if min(w, h) < 2:
            raise ValueError("image must be at least 2x2")
        cw, ch = max(1, round(w * self.config.crop_fraction)), max(1, round(h * self.config.crop_fraction))
        boxes = [(0, 0, w, h), (0, 0, cw, ch), (w - cw, 0, w, ch),
                 (0, h - ch, cw, h), (w - cw, h - ch, w, h)]
        size = self.clip.config.vision_config.image_size
        views = [image.crop(box).resize((size, size), Image.Resampling.BICUBIC) for box in boxes]
        pixels = self.processor.image_processor(images=views, do_resize=False, do_center_crop=False,
                                                return_tensors="pt").pixel_values.to(self.config.device)
        text = self.processor.tokenizer(phrase, return_tensors="pt", truncation=False).to(self.config.device)
        if text.input_ids.shape[1] > self.clip.config.text_config.max_position_embeddings:
            raise ValueError("visual phrase exceeds CLIP's text limit, pass a shorter --phrase")
        with torch.no_grad():
            embedding = F.normalize(self.clip.get_text_features(**text).float(), dim=-1)
        return pixels, embedding, boxes

    def forward(self, pixels, embedding):
        vision = self.clip.vision_model(pixel_values=pixels, output_hidden_states=self.config.patch_features == "value_residual")
        if self.config.patch_features == "value_residual":
            previous = vision.hidden_states[-2]
            layer = self.clip.vision_model.encoder.layers[-1]
            # Preserve local values before the last attention pooling and MLP.
            tokens = previous + layer.self_attn.out_proj(layer.self_attn.v_proj(layer.layer_norm1(previous)))
        else:
            tokens = vision.last_hidden_state
        tokens = tokens[:, 1:]
        tokens = self.clip.vision_model.post_layernorm(tokens)
        tokens = F.normalize(self.clip.visual_projection(tokens).float(), dim=-1)
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError("CLIP patch grid is not square")
        return (tokens @ embedding.T).squeeze(-1).reshape(len(pixels), side, side)

    @staticmethod
    def fuse(image, action, boxes):
        """Warp each view directly to image coordinates, preserving crop patch detail."""
        width, height = image.size
        if (action.ndim != 3 or len(action) != len(boxes) or not len(action)
                or not torch.isfinite(action).all() or boxes[0] != (0, 0, width, height)):
            raise ValueError("invalid spotlight action")
        fused = F.interpolate(action[:1, None], size=(height, width),
                              mode="bilinear", align_corners=False)[0, 0]
        for grid, (left, top, right, bottom) in zip(action[1:], boxes[1:]):
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError("crop box lies outside the image")
            local = F.interpolate(grid[None, None], size=(bottom - top, right - left),
                                  mode="bilinear", align_corners=False)[0, 0]
            fused[top:bottom, left:right] = torch.maximum(fused[top:bottom, left:right], local)
        return fused

    @torch.no_grad()
    def calibrate(self, image, reference, boxes):
        if self.config.mask_calibration == "none":
            return {"center": 0., "scale": 1.}
        fused = self.fuse(image, reference, boxes).float()
        # Fixed for the whole episode: candidates cannot shift their own normalization.
        return {"center": float(fused.mean()),
                "scale": max(float(fused.std(unbiased=False)), self.config.noise_std)}

    @torch.no_grad()
    def render(self, image, action, boxes, calibration=None):
        """Noise precedes fusion; reference statistics precede all candidate scoring."""
        if calibration is None:
            if self.config.mask_calibration != "none":
                raise ValueError("reference mask calibration is required")
            calibration = {"center": 0., "scale": 1.}
        center, scale = calibration["center"], calibration["scale"]
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid mask calibration")
        fused = self.fuse(image, action, boxes)
        mask = torch.sigmoid((fused - center) / (scale * self.config.mask_temperature)).cpu()
        original = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).float() / 255
        modified = original * (self.config.background_brightness + (1 - self.config.background_brightness) * mask[..., None])
        pixels = (modified.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(pixels), Image.fromarray((mask.numpy() * 255).round().astype(np.uint8))
