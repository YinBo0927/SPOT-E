"""Frozen Hugging Face VLM decoding and aligned full-vocabulary entropy readout."""
from dataclasses import dataclass
import inspect
import re

import torch

from .rewards import token_entropy

PROMPT_SUFFIX = (
    "\nRespond with exactly one line: Final answer: <answer>. "
    "Use only the requested word, number, option letter, yes/no, or short phrase as the answer. "
    "Do not include reasoning, a sentence, or a repetition of the question."
)


@dataclass
class BaselineTrace:
    text: str
    token_ids: torch.Tensor
    entropies: torch.Tensor
    answer_indices: torch.Tensor
    span_method: str


def extract_answer(text):
    markers = list(re.finditer(r"final\s+answer\s*:\s*", text, re.IGNORECASE))
    return text[markers[-1].end():].strip() if markers else text.strip()


def answer_span(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    matches = list(re.finditer(r"final\s+answer\s*:\s*", text, re.IGNORECASE))
    if matches:
        start = matches[-1].end()
        method = "final_answer_marker"
    else:
        # Some instruction-tuned models return only the short answer despite the template.
        start, method = 0, "whole_output_fallback"
    if not text[start:].strip():
        raise ValueError("VLM returned an empty final answer, increase --max-new-tokens")
    positions = []
    for i in range(len(token_ids)):
        prefix = tokenizer.decode(token_ids[:i + 1], skip_special_tokens=True)
        if len(prefix) > start:
            # Keep the entire suffix, including multi-token Unicode byte sequences.
            positions = list(range(i, len(token_ids)))
            break
    if not positions:
        raise ValueError("VLM returned an empty final answer, increase --max-new-tokens")
    return text, torch.tensor(positions, dtype=torch.long), method


class FrozenVLM:
    def __init__(self, model, processor, config):
        self.model = model.eval().requires_grad_(False)
        self.processor, self.config = processor, config

    @classmethod
    def from_pretrained(cls, config):
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        dtype = torch.bfloat16 if config.precision == "bf16" else torch.float32
        model_config = AutoConfig.from_pretrained(config.vlm_model)
        processor_kwargs = {}
        if model_config.model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
            processor_kwargs["max_pixels"] = config.max_pixels
        processor = AutoProcessor.from_pretrained(config.vlm_model, use_fast=False, **processor_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(config.vlm_model,
                    dtype=dtype, attn_implementation="eager").to(config.device)
        return cls(model, processor, config)

    def inputs(self, image, question, structured=True):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty")
        instruction = question + (PROMPT_SUFFIX if structured else "")
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": instruction}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                    add_generation_prompt=True, tokenize=False)
        kwargs = {"images": [image]} if image is not None else {}
        batch = self.processor(text=prompt, return_tensors="pt", **kwargs).to(self.config.device)
        # Qwen's processor may add text token types absent from the VLM forward API.
        batch.pop("token_type_ids", None)
        return batch

    @torch.inference_mode()
    def generate(self, image, question, structured=True, max_new_tokens=None):
        batch = self.inputs(image, question, structured)
        limit = max_new_tokens or self.config.max_new_tokens
        sequence = self.model.generate(**batch, do_sample=False, max_new_tokens=limit)
        tokens = sequence[0, batch.input_ids.shape[1]:].cpu()
        specials = set(self.processor.tokenizer.all_special_ids)
        if structured and len(tokens) >= limit and int(tokens[-1]) not in specials:
            raise ValueError("VLM response was truncated, increase --max-new-tokens")
        while len(tokens) and int(tokens[-1]) in specials:
            tokens = tokens[:-1]
        if not len(tokens):
            raise ValueError("VLM generated no answer tokens")
        return tokens

    @torch.inference_mode()
    def entropies(self, image, question, baseline_tokens):
        batch = self.inputs(image, question)
        tokens = baseline_tokens[None].to(self.config.device)
        batch["input_ids"] = torch.cat((batch.input_ids, tokens), dim=1)
        batch["attention_mask"] = torch.cat((batch.attention_mask, torch.ones_like(tokens)), dim=1)
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            batch["logits_to_keep"] = len(baseline_tokens) + 1
        logits = self.model(**batch, use_cache=False).logits[0, -(len(baseline_tokens) + 1):-1]
        if len(logits) != len(baseline_tokens):
            raise ValueError("VLM logits do not align with baseline token prefixes")
        return token_entropy(logits).cpu()

    def baseline(self, image, question):
        tokens = self.generate(image, question)
        text, indices, method = answer_span(self.processor.tokenizer, tokens)
        return BaselineTrace(text, tokens, self.entropies(image, question, tokens), indices, method)

    def visual_phrase(self, question):
        prompt = ("Rewrite the following question as one short visual phrase containing its key entities "
                  "and attributes. Do not answer it. Return only the phrase.\nQuestion: " + question)
        tokens = self.generate(None, prompt, structured=False, max_new_tokens=32)
        return self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def predict(self, image, question):
        tokens = self.generate(image, question)
        text, indices, method = answer_span(self.processor.tokenizer, tokens)
        return {"text": text, "answer": extract_answer(text),
                "answer_span_method": method}
