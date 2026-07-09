from __future__ import annotations

import re

from .schema import ParsedProblem


LATEX_PATTERNS = (
    r"\\[a-zA-Z]+",
    r"\$[^$]+\$",
    r"\\\(",
    r"\\\[",
    r"\^",
    r"_\{?",
)


def normalize_text(text: str) -> str:
    """Normalize whitespace while keeping user-visible mathematical content."""

    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    compact_lines = [line for line in lines if line]
    return "\n".join(compact_lines).strip()


def looks_like_latex(*texts: str) -> bool:
    joined = "\n".join(texts)
    return any(re.search(pattern, joined) for pattern in LATEX_PATTERNS)


def parse_input(problem: str, solution: str) -> ParsedProblem:
    normalized_problem = normalize_text(problem)
    normalized_solution = normalize_text(solution)
    if not normalized_problem:
        raise ValueError("Problem statement is empty.")
    if not normalized_solution:
        raise ValueError("Solution is empty.")

    return ParsedProblem(
        problem=normalized_problem,
        solution=normalized_solution,
        contains_latex=looks_like_latex(normalized_problem, normalized_solution),
    )
