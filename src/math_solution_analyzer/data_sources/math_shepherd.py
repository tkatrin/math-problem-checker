from __future__ import annotations

from .common import make_row


def normalize_math_shepherd_record(record: dict) -> list[dict[str, str | int]]:
    """Best-effort adapter for Math-Shepherd-style process supervision records.

    Public mirrors of Math-Shepherd data may differ in field names, so this
    adapter accepts common variants: question/problem, steps/solution_steps, and
    labels/step_labels with boolean, numeric, or string labels.
    """

    problem = str(record.get("problem") or record.get("question") or "")
    steps = record.get("steps") or record.get("solution_steps") or []
    labels = record.get("labels") or record.get("step_labels") or []
    problem_id = str(record.get("id") or record.get("problem_id") or f"math-shepherd-{hash(problem)}")
    rows: list[dict[str, str | int]] = []
    for index, step in enumerate([str(item) for item in steps]):
        raw_label = labels[index] if index < len(labels) else "unknown"
        label = _map_label(raw_label)
        rows.append(
            make_row(
                problem_id=f"math-shepherd-{problem_id}",
                domain="math",
                problem=problem,
                previous_steps=[str(item) for item in steps[:index]],
                current_step=step,
                next_step=str(steps[index + 1]) if index + 1 < len(steps) else "",
                step_index=index + 1,
                label=label,
                error_type="none" if label == "correct" else "process_error",
                explanation="Math-Shepherd process-supervision label.",
                source="math_shepherd",
            )
        )
    return rows


def _map_label(raw_label: object) -> str:
    if raw_label in (True, 1, "1", "+", "+1", "correct", "positive"):
        return "correct"
    if raw_label in (False, -1, "-1", "-", "incorrect", "negative"):
        return "incorrect"
    return "suspicious"
