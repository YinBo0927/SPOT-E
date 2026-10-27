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
        tokens = self.clip.vision_model(pixel_values=pixels).last_hidden_state[:, 1:]
        tokens = self.clip.vision_model.post_layernorm(tokens)
        tokens = F.normalize(self.clip.visual_projection(tokens).float(), dim=-1)
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError("CLIP patch grid is not square")
        return (tokens @ embedding.T).squeeze(-1).reshape(len(pixels), side, side)

    @torch.no_grad()
    def render(self, image, action, boxes):
        """Gaussian noise is applied to view-level relevance grids before max fusion."""
        width, height = image.size
        if action.ndim != 3 or len(action) != len(boxes) or not torch.isfinite(action).all():
            raise ValueError("invalid spotlight action")
        side = action.shape[-1]
        fused = action[0].clone()
        for grid, (left, top, right, bottom) in zip(action[1:], boxes[1:]):
            left, top = round(left * side / width), round(top * side / height)
            right, bottom = max(left + 1, round(right * side / width)), max(top + 1, round(bottom * side / height))
            local = F.interpolate(grid[None, None], size=(bottom - top, right - left),
                                  mode="bilinear", align_corners=False)[0, 0]
            fused[top:bottom, left:right] = torch.maximum(fused[top:bottom, left:right], local)
        fused = F.interpolate(fused[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        mask = torch.sigmoid(fused / self.config.mask_temperature).cpu()
        original = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).float() / 255
        modified = original * (self.config.background_brightness + (1 - self.config.background_brightness) * mask[..., None])
        pixels = (modified.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(pixels), Image.fromarray((mask.numpy() * 255).round().astype(np.uint8))
