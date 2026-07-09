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
    previous_step_count: int
    previous_context_chars: int
    has_final_answer: int
    answer_repeats_previous_number: int
    number_count: int
    variable_count: int
    domain_arithmetic: int
    domain_algebra: int
    domain_calculus: int
    domain_probability: int
    domain_linear_algebra: int
    domain_combinatorics: int
    sympy_equivalence_ok: int
    sympy_equivalence_error: int
    derivative_check_ok: int
    derivative_check_error: int
    linear_solution_check_error: int
    probability_fraction_warning: int

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
    joined = f"{problem} {' '.join(previous_steps)} {current_step}"
    domains = detect_domain(problem)
    equivalence = check_equation_equivalence(current_step)
    derivative = check_derivative(problem, current_step)
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
        previous_step_count=len(previous_steps),
        previous_context_chars=sum(len(step) for step in previous_steps),
        has_final_answer=int(any(marker in text for marker in ("ответ", "answer"))),
        answer_repeats_previous_number=int(_answer_repeats_previous_number(previous_steps, current_step)),
        number_count=len(re.findall(r"-?\d+(?:[.,]\d+)?", current_step)),
        variable_count=len(set(re.findall(r"\b[a-zA-Z]\b|[а-яА-ЯёЁ]", current_step))),
        domain_arithmetic=int(domains == "arithmetic"),
        domain_algebra=int(domains == "algebra"),
        domain_calculus=int(domains == "calculus"),
        domain_probability=int(domains == "probability"),
        domain_linear_algebra=int(domains == "linear_algebra"),
        domain_combinatorics=int(domains == "combinatorics"),
        sympy_equivalence_ok=int(equivalence is True),
        sympy_equivalence_error=int(equivalence is False),
        derivative_check_ok=int(derivative is True),
        derivative_check_error=int(derivative is False),
        linear_solution_check_error=int(check_linear_equation_solution(problem, current_step) is False),
        probability_fraction_warning=int(check_fraction_probability(problem, current_step, previous_steps) is False),
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


def detect_domain(problem: str) -> str:
    lowered = problem.lower()
    if any(marker in lowered for marker in ("вероят", "урн", "шар", "p(", "probab")):
        return "probability"
    if any(marker in lowered for marker in ("производ", "интеграл", "предел", "lim", "sin", "cos", "∫")):
        return "calculus"
    if any(marker in lowered for marker in ("матриц", "det", "[[")):
        return "linear_algebra"
    if any(marker in lowered for marker in ("систем", "уравнен")):
        return "algebra"
    if any(marker in lowered for marker in ("выбрать", "способ", "сочетан")):
        return "combinatorics"
    if re.search(r"\d+\s*[+\-*/]\s*\d+", lowered):
        return "arithmetic"
    return "unknown"


def check_equation_equivalence(text: str) -> bool | None:
    if sp is None or "=" not in text:
        return None
    match = re.search(r"([A-Za-z0-9+\-*/^().\s]+)=\s*([A-Za-z0-9+\-*/^().\s]+)", text.replace("^", "**"))
    if not match:
        return None
    left, right = match.groups()
    try:
        return bool(sp.simplify(sp.sympify(left) - sp.sympify(right)) == 0)
    except Exception:
        return None


def check_derivative(problem: str, step: str) -> bool | None:
    if sp is None or "производ" not in problem.lower() and "derivative" not in problem.lower():
        return None
    x = sp.symbols("x")
    problem_expr = _extract_function_expression(problem)
    step_expr = _extract_rhs_expression(step)
    if not problem_expr or not step_expr:
        return None
    try:
        expected = sp.diff(sp.sympify(problem_expr), x)
        actual = sp.sympify(step_expr)
        return bool(sp.simplify(expected - actual) == 0)
    except Exception:
        return None


def check_linear_equation_solution(problem: str, step: str) -> bool | None:
    if sp is None or "уравнен" not in problem.lower() or "x" not in problem:
        return None
    match_problem = re.search(r"(-?\d*)x\s*([+-]\s*\d+)?\s*=\s*(-?\d+)", problem.replace(" ", ""))
    match_step = re.search(r"x\s*=\s*(-?\d+(?:[.,]\d+)?)", step.replace(" ", ""))
    if not match_problem or not match_step:
        return None
    coef_raw, bias_raw, rhs_raw = match_problem.groups()
    coef = int(coef_raw) if coef_raw not in ("", "+", "-") else int(coef_raw + "1" if coef_raw in ("+", "-") else 1)
    bias = int((bias_raw or "0").replace(" ", ""))
    rhs = int(rhs_raw)
    candidate = float(match_step.group(1).replace(",", "."))
    return abs(coef * candidate + bias - rhs) < 1e-9


def check_fraction_probability(problem: str, step: str, previous_steps: list[str]) -> bool | None:
    lowered = problem.lower()
    if "без возвращ" not in lowered and "without replacement" not in lowered:
        return None
    step_lowered = step.lower()
    if not any(marker in step_lowered for marker in ("втор", "second")):
        return None
    current_fractions = set(re.findall(r"\b\d+\s*/\s*\d+\b", step))
    previous_fractions = {
        fraction
        for previous in previous_steps
        for fraction in re.findall(r"\b\d+\s*/\s*\d+\b", previous)
    }
    if current_fractions and current_fractions & previous_fractions:
        return False
    return True


def _safe_eval_binary(left: str, op: str, right: str) -> float | None:
    if sp is None:
        return None
    expression = f"{left.replace(',', '.')} {op} {right.replace(',', '.')}"
    try:
        return float(sp.N(sp.sympify(expression)))
    except Exception:
        return None


def _answer_repeats_previous_number(previous_steps: list[str], current_step: str) -> bool:
    if not previous_steps or not any(marker in current_step.lower() for marker in ("ответ", "answer")):
        return False
    previous_numbers = set(re.findall(r"-?\d+(?:[.,]\d+)?", " ".join(previous_steps)))
    current_numbers = set(re.findall(r"-?\d+(?:[.,]\d+)?", current_step))
    return bool(previous_numbers & current_numbers)


def _extract_function_expression(problem: str) -> str | None:
    normalized = problem.replace("^", "**").replace("\\(x\\)", "x")
    match = re.search(r"f\(x\)\s*=\s*([^.\n]+)", normalized)
    if not match:
        return None
    expr = match.group(1)
    return expr


def _extract_rhs_expression(step: str) -> str | None:
    normalized = step.replace("^", "**").replace("\\(x\\)", "x")
    match = re.search(r"f'\(x\)\s*=\s*([^.\n]+)", normalized)
    if not match:
        return None
    expr = match.group(1)
    return expr
