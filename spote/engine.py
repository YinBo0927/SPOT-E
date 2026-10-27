"""Independent per-image GRPO episodes with guaranteed LoRA reset."""
import time

from PIL import Image, ImageOps
import torch

from .config import SPOTEConfig
from .rewards import entropy_reward, gaussian_log_prob, grpo_loss
from .spotlight import SpotlightPolicy
from .vlm import FrozenVLM, extract_answer


class SPOTE:
    def __init__(self, vlm, spotlight, config):
        self.vlm, self.spotlight, self.config = vlm, spotlight, config

    @classmethod
    def from_pretrained(cls, config=None):
        config = config or SPOTEConfig()
        return cls(FrozenVLM.from_pretrained(config), SpotlightPolicy.from_pretrained(config), config)

    def __call__(self, image, question, phrase=None):
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        image = ImageOps.exif_transpose(image).convert("RGB")
        self.spotlight.reset()
        try:
            return self._episode(image, question, phrase)
        finally:
            # An exception during scoring must not leak adaptation into the next sample.
            self.spotlight.reset()

    def _episode(self, image, question, phrase):
        started, c = time.monotonic(), self.config
        trace = self.vlm.baseline(image, question)
        anchors = trace.entropies.argsort(stable=True)[:min(c.anchors, len(trace.entropies))]
        baseline_answer = extract_answer(trace.text)
        report = {"question": question, "baseline": trace.text, "baseline_answer": baseline_answer,
                  "baseline_answer_entropy": float(trace.entropies[trace.answer_indices].mean()),
                  "answer_span_method": trace.span_method, "anchor_indices": anchors.tolist(),
                  "settings": c.to_dict(), "iterations": [], "scoring": "baseline-prefix full-vocabulary entropy"}
        if hasattr(self.vlm, "model"):
            report["vlm_frozen"] = all(not p.requires_grad for p in self.vlm.model.parameters())
        if c.steps == 0:
            report.update(text=trace.text, answer=baseline_answer, selected_reward=0., elapsed_seconds=time.monotonic() - started)
            return {"image": image.copy(), "mask": Image.new("L", image.size, 255), "result": report}
        phrase = phrase if phrase is not None else self.vlm.visual_phrase(question)
        report["phrase"] = phrase
        pixels, text, boxes = self.spotlight.prepare(image, phrase)
        parameters = self.spotlight.trainable_parameters()
        optimizer = torch.optim.AdamW(parameters, lr=c.learning_rate, weight_decay=0.)
        rng = torch.Generator(device=c.device).manual_seed(c.seed)
        with torch.no_grad():
            reference = self.spotlight(pixels, text).detach().clone()
            calibration = self.spotlight.calibrate(image, reference, boxes)
        report["mask_calibration"] = calibration

        def sample(mean):
            return mean[None] + torch.randn((c.group_size, *mean.shape), device=mean.device, generator=rng) * c.noise_std

        def score(actions):
            images, masks, metrics = [], [], []
            for action in actions:
                modified, mask = self.spotlight.render(image, action, boxes, calibration)
                entropies = self.vlm.entropies(modified, question, trace.token_ids)
                metrics.append(entropy_reward(trace.entropies, entropies, trace.answer_indices, anchors,
                                              c.clarity_constant, c.preservation_weight))
                images.append(modified)
                masks.append(mask)
            rewards = torch.tensor([m["reward"] for m in metrics], device=c.device)
            if not torch.isfinite(rewards).all():
                raise ValueError("nonfinite VLM reward")
            return rewards, images, masks, metrics

        for step in range(c.steps):
            with torch.no_grad():
                old_mean = self.spotlight(pixels, text)
                actions = sample(old_mean).detach()
                old_logp = gaussian_log_prob(actions, old_mean, c.noise_std)
            rewards, _, _, metrics = score(actions)
            for _ in range(c.policy_epochs):
                optimizer.zero_grad(set_to_none=True)
                mean = self.spotlight(pixels, text)
                loss, kl = grpo_loss(mean, actions, rewards, old_logp, reference,
                                    c.noise_std, c.clip_range, c.kl_weight)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite GRPO loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
            report["iterations"].append({"step": step + 1, "loss": float(loss.detach()),
                                         "kl": float(kl.detach()), "gradient_norm": float(gradient_norm),
                                         "reward_std": float(rewards.std(unbiased=False)), "candidates": metrics})
        with torch.no_grad():
            actions = sample(self.spotlight(pixels, text))
        rewards, images, masks, metrics = score(actions)
        selected = int(rewards.argmax())
        prediction = self.vlm.predict(images[selected], question)
        change = max(float((p.detach() - self.spotlight.initial[name].to(p)).abs().max())
                     for name, p in self.spotlight.named_parameters() if name in self.spotlight.initial)
        report.update(prediction, selected_index=selected, selected_reward=metrics[selected]["reward"],
                      final_candidates=metrics, lora_max_parameter_change=change,
                      candidate_evaluations=(c.steps + 1) * c.group_size,
                      elapsed_seconds=time.monotonic() - started)
        return {"image": images[selected], "mask": masks[selected], "result": report}
