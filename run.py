"""Run one SPOT-E episode and export the answer, spotlight and mask."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--phrase", help="optional compact visual phrase, otherwise extracted by the frozen VLM")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs/default.json"))
    parser.add_argument("--output", default="outputs/example")
    for name in ("vlm-model", "clip-model", "device", "precision"):
        parser.add_argument("--" + name)
    for name in ("steps", "group-size", "max-new-tokens", "max-pixels", "seed"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory")
    settings = json.loads(Path(args.config).read_text())
    device = args.device or settings.get("device", "cuda")
    if device == "cpu":
        for variable in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            os.environ[variable] = ""
    import torch
    from PIL import Image
    from spote import SPOTE, SPOTEConfig
    torch.set_num_threads(args.threads)
    overrides = {key: getattr(args, key) for key in ("vlm_model", "clip_model", "device", "precision", "steps",
                                                   "group_size", "max_new_tokens", "max_pixels", "seed")}
    if device == "cpu" and args.precision is None:
        overrides["precision"] = "fp32"
    config = SPOTEConfig.load(args.config, **overrides)
    with Image.open(args.image) as source:
        image = source.copy()
    model = SPOTE.from_pretrained(config)
    result = model(image, args.question, phrase=args.phrase)
    output.mkdir(parents=True, exist_ok=True)
    result["image"].save(output / "spotlight.png")
    result["mask"].save(output / "mask.png")
    report = json.dumps(result["result"], indent=2, allow_nan=False) + "\n"
    (output / "result.json").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
