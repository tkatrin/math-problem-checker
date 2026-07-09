from __future__ import annotations

import re

from .schema import SolutionStep


STEP_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"(?:step|шаг)\s*\d+"
    r"|\d+[\).:-]"
    r"|[-*•]"
    r")\s*",
    flags=re.IGNORECASE,
)

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])")


def _split_by_markers(solution: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for line in solution.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        has_marker = bool(STEP_MARKER_RE.match(stripped))
        if has_marker and current:
            chunks.append(" ".join(current).strip())
            current = []
        current.append(STEP_MARKER_RE.sub("", stripped).strip() if has_marker else stripped)

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_by_connectors(solution: str) -> list[str]:
    normalized = " ".join(line.strip() for line in solution.splitlines() if line.strip())
    connector_pattern = re.compile(
        r"\s+(?=(?:then|therefore|hence|so|далее|затем|следовательно|значит|отсюда|поэтому)\b)",
        flags=re.IGNORECASE,
    )
    chunks = connector_pattern.split(normalized)
    if len(chunks) > 1:
        return [chunk.strip(" .") for chunk in chunks if chunk.strip(" .")]
    return [chunk.strip() for chunk in SENTENCE_BOUNDARY_RE.split(normalized) if chunk.strip()]


def split_solution_into_steps(solution: str) -> list[SolutionStep]:
    chunks = _split_by_markers(solution)
    if len(chunks) <= 1:
        chunks = _split_by_connectors(solution)

    return [SolutionStep(index=index, text=chunk) for index, chunk in enumerate(chunks, start=1)]
