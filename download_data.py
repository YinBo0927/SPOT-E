"""Export a pinned Hugging Face image QA dataset to a portable JSONL manifest."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--dataset", default="lmms-lab/ChartQA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--count", type=int, help="optional sample limit, otherwise export the entire split")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("count must be positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty dataset output directory")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from PIL import Image, ImageOps
    revision = HfApi().repo_info(args.dataset, repo_type='dataset', revision=args.revision, timeout=30).sha
    dataset = load_dataset(args.dataset, name=args.subset, split=args.split, revision=revision)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(itertools.islice(dataset, args.count)):
        image = row[args.image_column]
        if not isinstance(image, Image.Image):
            raise ValueError("image column must contain decoded PIL images")
        question = row[args.question_column]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("dataset contains an empty or invalid question")
        answers = row[args.answer_column]
        answers = answers if isinstance(answers, list) else [answers]
        if not answers or any(answer is None for answer in answers):
            raise ValueError("dataset contains missing answers")
        relative = f'images/{i:07d}.png'
        path = output / relative
        ImageOps.exif_transpose(image).convert('RGB').save(path)
        records.append({'id': str(i), 'image': relative, 'question': question,
                        'answers': [str(a) for a in answers], 'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f'exported {i + 1}', flush=True)
    if not records:
        raise ValueError("dataset split is empty")
    (output / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    (output / 'source.json').write_text(json.dumps({**vars(args), 'revision': revision,
        'samples': len(records), 'endpoint': os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}, indent=2) + '\n')
    print(output / 'samples.jsonl')


if __name__ == '__main__':
    main()
