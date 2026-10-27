# Implementation notes

This release implements arXiv:2606.20244v2, Sections 3–4. It does not include task-specific pretrained adapters. Each instance starts from zero-output visual LoRA, runs its own optimization, and resets the adapters even if evaluation raises an exception.

## Paper-defined components

- CLIP patch/text cosine similarities, global/crop max fusion, sigmoid masks and background dimming follow Eqs. (10)–(13).
- Both answer and anchor entropies use full-vocabulary logits under the baseline token prefixes, as specified in Section 3.1. The final answer is decoded separately on the selected image.
- Low-entropy anchors are selected over the baseline output. Only positive entropy increases incur the preservation penalty. Dynamic clarity and preservation follow Eqs. (9), (14)–(17).
- Group advantages use population standard deviation. GRPO uses clipped Gaussian likelihood ratios and an analytical KL to the initial policy, Eqs. (18)–(21).
- Default CLIP ViT-B/16, 8 update rounds, group size 4, LoRA rank 16, 60 anchors, preservation weight 0.5 and learning rate 5e-4 follow the paper's budget discussion and appendix tables. These defaults are not a claim that benchmark scores have been reproduced.

## Explicit implementation choices

The paper does not specify all details needed for an executable release. They are exposed in `configs/default.json`.

| Detail | This implementation |
| --- | --- |
| Visual phrase | Frozen VLM rewrites the question, or the caller supplies `phrase` |
| Views | Full image and four corner crops with side fraction 0.6 |
| CLIP geometry | Each view is resized to the CLIP grid without a hidden center crop |
| Fusion geometry | Crop maps are placed on the global patch grid before max fusion and final bilinear upsampling |
| Patch projection | Final visual LayerNorm and frozen CLIP visual projection |
| LoRA targets | Visual attention Q, K, V and output projections, alpha 16 |
| Stochastic policy | Independent Gaussian noise on each view's patch relevance, standard deviation 0.05 |
| Mask/background | Temperature 0.1, background brightness 0.3 |
| Reward/GRPO | Clarity constant 1, clip range 0.2, KL weight 0.01, AdamW without weight decay |
| Update reuse | One optimizer step per sampled group, configurable with `policy_epochs` |
| Selection | A fresh Best-of-4 group after the last update |

Gaussian log likelihoods and KL sum over all view/patch action dimensions. Log likelihood ratios are limited to ±20 before exponentiation to avoid overflow. Candidate sampling is independent of the current parameters during the policy gradient update. VLM scoring is outside autograd.

The requested output is one line, `Final answer: <short answer>`, without reasoning or an explanatory sentence. If the model returns only an answer, the complete output is used and `whole_output_fallback` is recorded. Empty or length-truncated answers raise an error. For outputs shorter than 60 tokens, all available positions are anchors.

The local backend uses Hugging Face `AutoModelForImageTextToText` and processor chat templates. Models that require custom remote chat code need a separate adapter. Proprietary APIs exposing only truncated top-k log probabilities are not treated as exact full-vocabulary entropy sources.

For another backend, pass it to `SPOTE(vlm, spotlight, config)`. It supplies `baseline(image, question)` returning a `BaselineTrace`, `entropies(image, question, baseline_tokens)`, `visual_phrase(question)` and `predict(image, question)`. The entropy method must score all baseline token positions without resampling their prefixes.

`evaluate.py` supports string exact match and ChartQA relaxed accuracy. It is not a replacement for the official evaluation tools of every benchmark in the paper. Labels are used only for scoring completed predictions.

## Validation scope

CPU execution checks use real Qwen2.5-VL-7B and CLIP ViT-B/16 weights at a reduced VLM image budget of 50,176 pixels. Checks cover the 8-round, group-of-4 optimization budget, a natural-photo example, dataset export, paired evaluation, and resume. Separate numerical checks cover reward equations, Gaussian GRPO, adapter-only gradients, crop placement, Unicode answer spans, truncated-output rejection and reset after both successful and failed episodes.

CUDA execution, full-resolution benchmark scores and paper runtimes have not been verified by these checks. The README's result figures are from the paper.
