"""Basic exact match and ChartQA's 5% relaxed accuracy, not a benchmark harness."""
import math


def score_answer(prediction, answers, metric="relaxed"):
    if not isinstance(answers, list) or not answers:
        raise ValueError("answers must be a nonempty list of accepted strings")
    if metric not in ("exact", "relaxed"):
        raise ValueError("unknown metric")
    prediction = str(prediction).strip().casefold()

    def number(text):
        try:
            value = float(text[:-1]) / 100 if text.endswith("%") else float(text)
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    for answer in answers:
        target = str(answer).strip().casefold()
        if prediction == target:
            return 1.
        p, t = number(prediction), number(target)
        if metric == "relaxed" and p is not None and t is not None:
            if abs(p - t) <= .05 * abs(t):
                return 1.
    return 0.
