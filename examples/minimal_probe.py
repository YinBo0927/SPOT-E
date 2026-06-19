"""Minimal SPOT-E wiring example with a fake frozen VLM probe.

Replace FakeProbe.trace with an adapter around Qwen/LLaVA/InternVL or a VLM API
that returns token-level entropies and the final-answer token span.
"""

from PIL import Image

from spot_e import SPOTEPlugin, TokenTrace


class FakeProbe:
    def trace(self, image: Image.Image, question: str, *, baseline_trace=None) -> TokenTrace:
        del image, question, baseline_trace
        return TokenTrace(
            tokens=["Final", "answer", ":", "red"],
            entropies=[0.05, 0.07, 0.02, 0.42],
            answer_span=(3, 4),
            text="Final answer: red",
        )


image = Image.new("RGB", (224, 224), color="white")
plugin = SPOTEPlugin(FakeProbe(), seed=7)
result = plugin.run(image, "What color is the upper garment?")

print(result.selected_evaluation)
print(result.anchors)
