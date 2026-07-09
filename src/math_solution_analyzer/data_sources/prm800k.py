from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .common import iter_json_records, make_row, write_rows_csv


RATING_TO_LABEL = {
    1: ("correct", "none", "Human label: step is correct/progress-making."),
    0: ("suspicious", "no_progress", "Human label: step is not incorrect, but makes little or no progress."),
    -1: ("incorrect", "process_error", "Human label: step is incorrect."),
}


def normalize_prm800k_record(record: dict, *, selected_only: bool = True) -> list[dict[str, str | int]]:
    """Convert one PRM800K labeled solution record into step-level rows.

    PRM800K stores one full trajectory per JSONL line. Each trajectory contains
    a MATH problem and step-level completions with human ratings -1, 0, +1.
    """

    question = record.get("question") or {}
    problem = str(question.get("problem") or "")
    problem_id = str(question.get("problem_id") or question.get("id") or record.get("timestamp") or "prm800k-record")
    steps = ((record.get("label") or {}).get("steps") or [])
    rows: list[dict[str, str | int]] = []
    previous_steps: list[str] = []

    for step_index, step in enumerate(steps, start=1):
        completions = step.get("completions") or []
        chosen_completion = step.get("chosen_completion")
        human_completion = step.get("human_completion")

        selected_completions: list[dict]
        if selected_only:
            if human_completion and chosen_completion is None:
                selected_completions = [{"text": human_completion, "rating": 1}]
            elif isinstance(chosen_completion, int) and 0 <= chosen_completion < len(completions):
                selected_completions = [completions[chosen_completion]]
            elif completions:
                selected_completions = [completions[0]]
            else:
                selected_completions = []
        else:
            selected_completions = completions

        next_step = _peek_next_step_text(steps, step_index)
        for completion in selected_completions:
            text = str(completion.get("text") or "").strip()
            if not text:
                continue
            rating = int(completion.get("rating") or 0)
            label, error_type, explanation = RATING_TO_LABEL.get(rating, RATING_TO_LABEL[0])
            rows.append(
                make_row(
                    problem_id=f"prm800k-{problem_id}",
                    domain="math",
                    problem=problem,
                    previous_steps=previous_steps,
                    current_step=text,
                    next_step=next_step,
                    step_index=step_index,
                    label=label,
                    error_type=error_type,
                    explanation=explanation,
                    source="prm800k",
                )
            )
        if selected_completions:
            previous_steps.append(str(selected_completions[0].get("text") or "").strip())
    return rows


def read_prm800k_jsonl(path: Path, *, selected_only: bool = True, limit: int | None = None) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for index, record in enumerate(iter_json_records(path)):
        if limit is not None and index >= limit:
            break
        rows.extend(normalize_prm800k_record(record, selected_only=selected_only))
    return rows


def _peek_next_step_text(steps: list[dict], current_step_index: int) -> str:
    if current_step_index >= len(steps):
        return ""
    next_step = steps[current_step_index]
    completions = next_step.get("completions") or []
    chosen = next_step.get("chosen_completion")
    if isinstance(chosen, int) and 0 <= chosen < len(completions):
        return str(completions[chosen].get("text") or "")
    if completions:
        return str(completions[0].get("text") or "")
    return str(next_step.get("human_completion") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/prm800k_steps.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all-completions", action="store_true")
    args = parser.parse_args()
    rows = read_prm800k_jsonl(args.input, selected_only=not args.all_completions, limit=args.limit)
    write_rows_csv(rows, args.output)
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
