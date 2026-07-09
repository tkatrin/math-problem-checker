from __future__ import annotations

import argparse
import sys
from pathlib import Path

from math_solution_analyzer import analyze_solution
from math_solution_analyzer.models.predict import StepMLClassifier


def main() -> None:
    parser = argparse.ArgumentParser(prog="math-checker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Analyze a math solution and print structured JSON.")
    check_parser.add_argument("--problem", type=Path, required=True)
    check_parser.add_argument("--solution", type=Path, required=True)
    check_parser.add_argument("--model", type=Path, default=None, help="Path to a trained .joblib ML model.")
    check_parser.add_argument("--output", type=Path, default=None, help="Optional path for JSON output.")
    check_parser.add_argument("--no-ml", action="store_true", help="Disable ML classifier and use rule-based checks only.")
    check_parser.add_argument("--no-llm", action="store_true", help="Disable optional OpenAI explanation.")

    args = parser.parse_args()
    if args.command == "check":
        problem = args.problem.read_text(encoding="utf-8")
        solution = args.solution.read_text(encoding="utf-8")
        ml_classifier = None
        if args.model is not None and not args.no_ml:
            ml_classifier = StepMLClassifier(args.model)
        report = analyze_solution(
            problem,
            solution,
            use_llm=not args.no_llm,
            use_ml=not args.no_ml,
            ml_classifier=ml_classifier,
        )
        output = report.model_dump_json(indent=2)
        if args.output is not None:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        if report.metadata.get("llm_warning"):
            print(str(report.metadata["llm_warning"]), file=sys.stderr)


if __name__ == "__main__":
    main()
