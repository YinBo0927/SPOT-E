# Implementation notes

This release implements arXiv:2606.20244v2, Sections 3–4. It does not include task-specific pretrained adapters. Each instance starts from zero-output visual LoRA, runs its own optimization, and resets the adapters even if evaluation raises an exception.

## Paper-defined components

- CLIP patch/text cosine similarities, global/crop max fusion, sigmoid masks and background dimming implement the structure of Eqs. (10)–(13). The default adds the reference calibration described below.
- Both answer and anchor entropies use full-vocabulary logits under the baseline token prefixes, as specified in Section 3.1. The final answer is decoded separately on the selected image.
- Low-entropy anchors are selected over the baseline output. Only positive entropy increases incur the preservation penalty. Dynamic clarity and preservation follow Eqs. (9), (14)–(17).
- Group advantages use population standard deviation. GRPO uses clipped Gaussian likelihood ratios and an analytical KL to the initial policy, Eqs. (18)–(21).
- Default CLIP ViT-B/16, 8 update rounds, group size 4, LoRA rank 16, 60 anchors, preservation weight 0.5 and learning rate 5e-4 follow the paper's budget discussion and appendix tables. These defaults are not a claim that benchmark scores have been reproduced.

## Explicit implementation choices

The paper does not specify all details needed for an executable release. They are exposed in `configs/default.json`.

| Detail | This implementation |
| --- | --- |
| Visual phrase | Frozen VLM copies a contiguous phrase from the question. Unverifiable rewrites fall back to the original question. The caller can supply `phrase` |
| Views | Full image and four corner crops with side fraction 0.6 |
| CLIP geometry | Each view is resized to the CLIP grid without a hidden center crop |
| Fusion geometry | Each map is bilinearly resized directly into its original image box, then max-fused at image resolution. Crop detail is not downsampled to the global patch grid |
| Patch features | Last-layer value and output projections with the preceding residual, followed by visual LayerNorm and the frozen CLIP projection. Set `patch_features` to `last_hidden` for the previous full-transformer output |
| LoRA targets | Visual attention Q, K, V and output projections, alpha 16 |
| Stochastic policy | Independent Gaussian noise on each view's patch relevance, standard deviation 0.01 |
| Mask/background | Fixed reference calibration, temperature 0.7, background brightness 0.2 |
| Reward/GRPO | Clarity constant 1, clip range 0.2, KL weight 0.01, AdamW without weight decay |
| Update reuse | One optimizer step per sampled group, configurable with `policy_epochs` |
| Selection | A fresh Best-of-4 group after the last update |

Gaussian log likelihoods and KL sum over all view/patch action dimensions. Log likelihood ratios are limited to ±20 before exponentiation to avoid overflow. Candidate sampling is independent of the current parameters during the policy gradient update. VLM scoring is outside autograd.

The default dense feature path bypasses the last attention mixing and MLP to retain local information. It follows the value/residual extraction idea from [MaskCLIP](https://github.com/chongzhou96/MaskCLIP); it is an implementation choice rather than a paper-specified SPOT-E setting. The CLIP encoder and text encoder remain frozen apart from the visual LoRA adapters. Lower sampling noise and the mask defaults were chosen for clearer spotlights in the natural-image examples, not to claim reproduced benchmark results.

Raw positive cosine scores made the previous masks saturate near one. With `mask_calibration: "reference"`, the initial, noise-free fused map supplies a spatial mean `center` and `scale = max(spatial_std, noise_std)`. Every candidate uses `sigmoid((fused - center) / (scale * mask_temperature))`. These statistics remain fixed throughout the episode and are saved in `result.json`. The noise floor also handles uniform maps without division by zero. This calibration is an implementation choice, not a reported paper hyperparameter. Set `mask_calibration: "none"` and `mask_temperature: 0.1` for the raw-cosine sigmoid, while retaining the corrected crop geometry.

The requested output is one line, `Final answer: <short answer>`, without reasoning or an explanatory sentence. If the model returns only an answer, the complete output is used and `whole_output_fallback` is recorded. Empty or length-truncated answers raise an error. For outputs shorter than 60 tokens, all available positions are anchors.

The local backend uses Hugging Face `AutoModelForImageTextToText` and processor chat templates. Models that require custom remote chat code need a separate adapter. Proprietary APIs exposing only truncated top-k log probabilities are not treated as exact full-vocabulary entropy sources.

For another backend, pass it to `SPOTE(vlm, spotlight, config)`. It supplies `baseline(image, question)` returning a `BaselineTrace`, `entropies(image, question, baseline_tokens)`, `visual_phrase(question)` and `predict(image, question)`. The entropy method must score all baseline token positions without resampling their prefixes.

`evaluate.py` supports string exact match and ChartQA relaxed accuracy. It is not a replacement for the official evaluation tools of every benchmark in the paper. Labels are used only for scoring completed predictions.

## Validation scope

CPU execution checks use real Qwen2.5-VL-7B and CLIP ViT-B/16 weights at a reduced VLM image budget of 50,176 pixels. Checks cover the 8-round, group-of-4 optimization budget, a natural-photo example, dataset export, paired evaluation, and resume. Separate numerical checks cover reward equations, Gaussian GRPO, adapter-only gradients, crop placement, Unicode answer spans, truncated-output rejection and reset after both successful and failed episodes.

GPU checks additionally run the real models in BF16 with an 802,816-pixel VLM budget, eight rounds and four candidates per group. Three photographs were used to choose the feature and mask settings. A fourth photograph was then tested with two questions targeting different regions. The checks cover 36 candidate evaluations per image, unchanged VLM weights, adapter reset and the command-line entry point. Approximate region rectangles are used only for inspecting results. Separate regression checks cover every patch position in all four crop views, constant-map calibration, rejection of invented phrase attributes and gradients through the dense visual LoRA path. These checks establish execution and rendering behavior, not benchmark accuracy gains. NVIDIA execution, full benchmark scores and paper runtimes remain unverified. The README's Paper Results figures are from the paper.
