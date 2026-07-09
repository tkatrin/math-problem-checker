from __future__ import annotations

import argparse
from pathlib import Path

from math_solution_analyzer import analyze_solution


def main() -> None:
    parser = argparse.ArgumentParser(prog="math-checker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Analyze a math solution and print structured JSON.")
    check_parser.add_argument("--problem", type=Path, required=True)
    check_parser.add_argument("--solution", type=Path, required=True)
    check_parser.add_argument("--no-llm", action="store_true", help="Disable optional OpenAI explanation.")

    args = parser.parse_args()
    if args.command == "check":
        problem = args.problem.read_text(encoding="utf-8")
        solution = args.solution.read_text(encoding="utf-8")
        report = analyze_solution(problem, solution, use_llm=not args.no_llm)
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
