"""Full-vocabulary entropy, baseline anchors and Eqs. (9), (14)-(21)."""
import math

import torch
from torch.nn import functional as F


def token_entropy(logits):
    """Natural-log Shannon entropy of raw, untruncated vocabulary logits."""
    logits = logits.float()
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("expected finite [tokens,vocabulary] logits")
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1)


def entropy_reward(baseline, candidate, answer_indices, anchor_indices, c=1., weight=.5):
    """Both trajectories are teacher-forced on the same baseline token prefixes."""
    if (baseline.ndim != 1 or candidate.shape != baseline.shape or not len(answer_indices)
            or not len(anchor_indices) or not torch.isfinite(candidate).all()
            or not torch.isfinite(baseline).all()):
        raise ValueError("aligned finite token entropies and nonempty spans are required")
    if c <= 0 or weight < 0:
        raise ValueError("invalid reward settings")
    base_answer = baseline[answer_indices].mean()
    answer = candidate[answer_indices].mean()
    disruption = (candidate[anchor_indices] - baseline[anchor_indices]).clamp_min(0).mean()
    clarity = base_answer / (base_answer + c) * (base_answer - answer)
    return {"reward": float(clarity - weight * disruption), "clarity": float(clarity),
            "anchor_disruption": float(disruption), "answer_entropy": float(answer)}


def gaussian_log_prob(actions, mean, std):
    if std <= 0 or actions.shape[1:] != mean.shape:
        raise ValueError("Gaussian action and policy dimensions differ")
    return (-.5 * ((actions - mean) / std).square() - math.log(std * math.sqrt(2 * math.pi))).flatten(1).sum(1)


def grpo_loss(mean, actions, rewards, old_log_prob, reference_mean, std, clip_range=.2, kl_weight=.01):
    """Joint Gaussian likelihood ratios and exact fixed-covariance Gaussian KL."""
    if rewards.ndim != 1 or len(rewards) != len(actions) or len(rewards) < 2:
        raise ValueError("GRPO requires a group of at least two scalar rewards")
    advantages = ((rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)).detach()
    log_ratio = gaussian_log_prob(actions.detach(), mean, std) - old_log_prob.detach()
    ratio = log_ratio.clamp(-20, 20).exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages)
    kl = .5 * ((mean - reference_mean.detach()) / std).square().sum()
    return -surrogate.mean() + kl_weight * kl, kl
