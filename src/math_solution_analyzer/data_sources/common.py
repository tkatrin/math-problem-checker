from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


PROJECT_COLUMNS = [
    "problem_id",
    "domain",
    "problem",
    "previous_steps",
    "current_step",
    "next_step",
    "step_index",
    "label",
    "error_type",
    "explanation",
    "source",
]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_json_records(path: Path) -> Iterable[dict]:
    """Yield records from either a JSONL file or a JSON array export."""

    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object records in {path}")
        yield record


def write_rows_csv(rows: Iterable[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROJECT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PROJECT_COLUMNS})


def make_row(
    *,
    problem_id: str,
    domain: str,
    problem: str,
    previous_steps: list[str],
    current_step: str,
    next_step: str = "",
    step_index: int,
    label: str,
    error_type: str,
    explanation: str,
    source: str,
) -> dict[str, str | int]:
    return {
        "problem_id": problem_id,
        "domain": domain,
        "problem": problem,
        "previous_steps": " ||| ".join(previous_steps),
        "current_step": current_step,
        "next_step": next_step,
        "step_index": step_index,
        "label": label,
        "error_type": error_type,
        "explanation": explanation,
        "source": source,
    }
