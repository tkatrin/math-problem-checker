from __future__ import annotations

from .checker import StepChecker, build_checker
from .parser import parse_input
from .report_generator import generate_report
from .schema import AnalysisReport, StepAnalysis
from .step_splitter import split_solution_into_steps


def analyze_solution(problem: str, solution: str, *, use_llm: bool = True, checker: StepChecker | None = None) -> AnalysisReport:
    parsed = parse_input(problem, solution)
    steps = split_solution_into_steps(parsed.solution)
    selected_checker = checker or build_checker(use_llm=use_llm)

    analyses: list[StepAnalysis] = []
    for step in steps:
        previous_steps = [analysis.step for analysis in analyses]
        analyses.append(
            selected_checker.check_step(
                problem=parsed.problem,
                previous_steps=previous_steps,
                current_step=step,
            )
        )

    return generate_report(parsed.problem, analyses, contains_latex=parsed.contains_latex)
