from __future__ import annotations

import argparse
from pathlib import Path

from .common import iter_jsonl, make_row, write_rows_csv


def normalize_processbench_record(record: dict, *, label_index_base: int = 0) -> list[dict[str, str | int]]:
    """Convert one ProcessBench case into project step rows.

    ProcessBench labels the earliest erroneous step. A label of -1 is treated
    as "all steps correct"; otherwise the label is interpreted as the error
    step index using label_index_base.
    """

    problem = str(record.get("problem") or "")
    steps = [str(step) for step in (record.get("steps") or [])]
    raw_label = int(record.get("label", -1))
    error_index = raw_label - label_index_base if raw_label >= 0 else -1
    problem_id = str(record.get("id") or f"processbench-{hash(problem)}")

    rows: list[dict[str, str | int]] = []
    for index, step in enumerate(steps):
        if error_index < 0 or index < error_index:
            label = "correct"
            error_type = "none"
            explanation = "ProcessBench: step occurs before the first annotated error."
        elif index == error_index:
            label = "incorrect"
            error_type = "process_error"
            explanation = "ProcessBench: earliest human-annotated erroneous step."
        else:
            label = "suspicious"
            error_type = "after_error_context"
            explanation = "ProcessBench: step follows the first annotated error and may be contaminated."
        rows.append(
            make_row(
                problem_id=f"processbench-{problem_id}",
                domain=str(record.get("split") or "math"),
                problem=problem,
                previous_steps=steps[:index],
                current_step=step,
                next_step=steps[index + 1] if index + 1 < len(steps) else "",
                step_index=index + 1,
                label=label,
                error_type=error_type,
                explanation=explanation,
                source="processbench",
            )
        )
    return rows


def read_processbench_jsonl(path: Path, *, label_index_base: int = 0, limit: int | None = None) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for index, record in enumerate(iter_jsonl(path)):
        if limit is not None and index >= limit:
            break
        rows.extend(normalize_processbench_record(record, label_index_base=label_index_base))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/processbench_steps.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label-index-base", type=int, default=0)
    args = parser.parse_args()
    rows = read_processbench_jsonl(args.input, label_index_base=args.label_index_base, limit=args.limit)
    write_rows_csv(rows, args.output)
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
