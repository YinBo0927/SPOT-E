"""How to plug another VLM into SPOT-E.

Edit this file when you want to connect Qwen-VL, LLaVA, InternVL, GPT-style
APIs, or any other VLM.

The only required integration point is `YourVLMProbe.trace(...)` below. SPOT-E
will call it many times:

1. once on the original image to get the baseline token entropies;
2. again on spotlight images, passing `baseline_trace`, so anchor positions can
   be scored on the same baseline token prefix.

Your adapter should keep the VLM frozen and return a `TokenTrace`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from spot_e import SPOTEPlugin, TokenTrace


class YourVLMProbe:
    """Replace the placeholder methods in this class with your model code."""

    def __init__(self) -> None:
        # Load your frozen VLM here.
        #
        # Examples:
        #   self.model = AutoModelForVision2Seq.from_pretrained(...)
        #   self.processor = AutoProcessor.from_pretrained(...)
        #   self.client = OpenAI(...)
        #
        # Do not enable training for the VLM. SPOT-E only adapts the spotlight.
        self.model = None
        self.processor = None

    def trace(
        self,
        image: Image.Image,
        question: str,
        *,
        baseline_trace: TokenTrace | None = None,
    ) -> TokenTrace:
        """Return generated/scored tokens, per-token entropy, and answer span.

        For the baseline call (`baseline_trace is None`):
            generate a structured answer, for example:
            "Reason briefly. Final answer: <answer>"

        For candidate spotlight calls (`baseline_trace is not None`):
            score the baseline token prefix under the new image when possible.
            This keeps low-entropy anchor positions aligned with the paper.
        """

        prompt = self._build_prompt(question)

        if baseline_trace is None:
            tokens, entropies, text = self._generate_with_entropies(image, prompt)
        else:
            tokens = list(baseline_trace.tokens)
            entropies = self._score_prefix_entropies(image, prompt, tokens)
            text = baseline_trace.text

        answer_span = self._find_final_answer_span(tokens, text)
        return TokenTrace(
            tokens=tokens,
            entropies=entropies,
            answer_span=answer_span,
            text=text,
        )

    def _build_prompt(self, question: str) -> str:
        return (
            f"{question}\n"
            "Answer with a short rationale, then write exactly: Final answer: ..."
        )

    def _generate_with_entropies(
        self,
        image: Image.Image,
        prompt: str,
    ) -> tuple[list[str], list[float], str]:
        """Run your VLM and collect next-token entropies.

        Implementation notes for local Hugging Face VLMs:
        - call `generate(..., output_scores=True, return_dict_in_generate=True)`;
        - decode generated token ids into `tokens`;
        - convert each score/logit vector to entropy;
        - decode the full text into `text`.

        Implementation notes for API VLMs:
        - request token logprobs/logits if the provider supports them;
        - compute entropy from the returned token distribution;
        - if only top-k logprobs are returned, document that entropy is
          approximate.
        """

        del image, prompt
        return (
            ["Reason", ".", "Final", "answer", ":", "red"],
            [0.32, 0.18, 0.05, 0.06, 0.03, 0.42],
            "Reason. Final answer: red",
        )

    def _score_prefix_entropies(
        self,
        image: Image.Image,
        prompt: str,
        tokens: list[str],
    ) -> list[float]:
        """Score an existing token prefix under `image` and return entropies.

        This is the preferred path for spotlight candidates because it aligns
        anchor token positions with the original baseline trace.
        """

        del image, prompt
        return [0.30 if token != "red" else 0.36 for token in tokens]

    def _find_final_answer_span(
        self,
        tokens: list[str],
        text: str,
    ) -> tuple[int, int]:
        """Return the half-open token span for the final answer.

        Replace this with tokenizer-aware span extraction for your model. The
        fallback below works for this toy example.
        """

        del text
        try:
            colon_index = tokens.index(":")
            return (colon_index + 1, len(tokens))
        except ValueError:
            return (max(0, len(tokens) - 1), len(tokens))


def main() -> None:
    image_path = Path("example.png")
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), color="white")

    probe = YourVLMProbe()
    plugin = SPOTEPlugin(probe, seed=7)
    result = plugin.run(image, "What color is the upper garment?")

    result.image.save("spotlight.png")
    print(result.selected_evaluation)
    print("anchors:", result.anchors)


if __name__ == "__main__":
    main()
