from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import sympy as sp
except Exception:  # pragma: no cover - optional dependency during import-time docs builds
    sp = None


ARITHMETIC_RE = re.compile(
    r"(?<!/)(-?\d+(?:[.,]\d+)?)\s*([+\-*/])\s*(-?\d+(?:[.,]\d+)?)\s*=\s*(-?\d+(?:[.,]\d+)?)(?!\s*/)"
)


@dataclass(frozen=True)
class StepFeatureRow:
    has_equality: int
    has_implication: int
    has_fraction: int
    formula_count: int
    token_count: int
    char_count: int
    step_index: int
    sympy_error: int
    contains_final_marker: int
    contains_probability_marker: int
    contains_derivative_marker: int
    contains_substitution_marker: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def extract_step_features(
    *,
    problem: str,
    previous_steps: list[str],
    current_step: str,
    step_index: int,
) -> dict[str, int]:
    text = current_step.lower()
    return StepFeatureRow(
        has_equality=int("=" in current_step),
        has_implication=int(any(marker in text for marker in ("=>", "следовательно", "значит", "therefore", "hence"))),
        has_fraction=int(bool(re.search(r"\b\d+\s*/\s*\d+\b|\\frac", current_step))),
        formula_count=len(re.findall(r"[=<>≤≥]|\\frac|\\sin|\\cos|\\lim|\^", current_step)),
        token_count=len(re.findall(r"\w+", current_step)),
        char_count=len(current_step),
        step_index=step_index,
        sympy_error=int(sympy_arithmetic_error(current_step)),
        contains_final_marker=int(any(marker in text for marker in ("ответ", "answer", "итак", "therefore"))),
        contains_probability_marker=int(any(marker in (problem + " " + current_step).lower() for marker in ("вероят", "probab", "урн", "шар"))),
        contains_derivative_marker=int(any(marker in text for marker in ("производ", "derivative", "f'", "sin", "cos"))),
        contains_substitution_marker=int(any(marker in text for marker in ("подстав", "substitut", "тогда"))),
    ).as_dict()


def make_model_text(problem: str, previous_steps: list[str], current_step: str) -> str:
    previous = " ".join(previous_steps[-3:]) if previous_steps else "Нет предыдущих шагов."
    return f"[PROBLEM] {problem} [PREVIOUS] {previous} [STEP] {current_step}"


def sympy_arithmetic_error(text: str) -> bool:
    for left, op, right, result in ARITHMETIC_RE.findall(text):
        expected = _safe_eval_binary(left, op, right)
        if expected is None:
            continue
        actual = float(result.replace(",", "."))
        if abs(expected - actual) > 1e-9:
            return True
    return False


def _safe_eval_binary(left: str, op: str, right: str) -> float | None:
    if sp is None:
        return None
    expression = f"{left.replace(',', '.')} {op} {right.replace(',', '.')}"
    try:
        return float(sp.N(sp.sympify(expression)))
    except Exception:
        return None
