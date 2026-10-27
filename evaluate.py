"""Matched baseline/SPOT-E evaluation on an image/question/answers JSONL."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs/default.json"))
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--metric", choices=("exact", "relaxed"), default="relaxed")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    device = args.device or settings.get("device", "cuda")
    if device == "cpu":
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            os.environ[key] = ""
    import torch
    from PIL import Image
    from spote import SPOTE, SPOTEConfig
    from spote.metrics import score_answer
    from spote.vlm import PROMPT_SUFFIX
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    config = SPOTEConfig.load(args.config, device=args.device,
        precision=args.precision or ("fp32" if device == "cpu" else None), steps=args.steps, group_size=args.group_size)
    manifest, output = Path(args.manifest).resolve(), Path(args.output).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError("manifest needs nonempty rows with unique IDs")
    for row in rows:
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError("question must be nonempty text")
        if not isinstance(row['answers'], list) or not row['answers']:
            raise ValueError("each row requires accepted answers for evaluation")
        if not (manifest.parent / row['image']).is_file():
            raise FileNotFoundError(row['image'])
    expected = {"config": config.to_dict(), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "metric": args.metric, "save_images": args.save_images, "prompt_suffix": PROMPT_SUFFIX}
    if args.resume:
        if json.loads((output / 'settings.json').read_text()) != expected:
            raise ValueError("resume configuration or manifest differs")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory, or --resume")
    (output / 'predictions').mkdir(parents=True, exist_ok=True)
    (output / 'settings.json').write_text(json.dumps(expected, indent=2) + '\n')
    model, records = None, []
    for i, row in enumerate(rows):
        path = output / 'predictions' / f'{i:07d}.json'
        image_path = manifest.parent / row['image']
        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if row.get('image_sha256', image_sha256) != image_sha256:
            raise ValueError("image differs from the dataset manifest")
        if args.resume and path.is_file():
            record = json.loads(path.read_text())
            if record['id'] != row['id'] or record['image_sha256'] != image_sha256:
                raise ValueError("saved prediction ID or source image differs")
        else:
            if model is None:
                model = SPOTE.from_pretrained(config)
            with Image.open(image_path) as source:
                image = source.copy()
            # Ground-truth answers are used below, never passed to adaptation.
            result = model(image, row['question'], phrase=row.get('phrase'))
            report = result['result']
            record = {'id': row['id'], 'image_sha256': image_sha256, **report,
                      'baseline_score': score_answer(report['baseline_answer'], row['answers'], args.metric),
                      'score': score_answer(report['answer'], row['answers'], args.metric)}
            if args.save_images:
                directory = output / 'images' / f'{i:07d}'
                directory.mkdir(parents=True, exist_ok=True)
                result['image'].save(directory / 'spotlight.png')
                result['mask'].save(directory / 'mask.png')
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record, allow_nan=False) + '\n')
            temporary.replace(path)
        records.append(record)
        print(f"evaluated or restored {i + 1}/{len(rows)}", flush=True)
    summary = {'samples': len(records), 'metric': args.metric,
               'baseline_score': sum(r['baseline_score'] for r in records) / len(records),
               'spote_score': sum(r['score'] for r in records) / len(records)}
    (output / 'predictions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
