from __future__ import annotations

from .schema import AnalysisReport, StepAnalysis, StepStatus


def generate_report(problem: str, steps: list[StepAnalysis], *, contains_latex: bool = False) -> AnalysisReport:
    correct: list[str] = []
    errors: list[str] = []
    missing: list[str] = []
    fixes: list[str] = []

    for analysis in steps:
        prefix = f"Шаг {analysis.step.index}: "
        correct.extend(prefix + item for item in analysis.what_is_correct)
        errors.extend(prefix + issue.explanation for issue in analysis.possible_errors)
        missing.extend(prefix + item for item in analysis.missing_steps)
        fixes.extend(prefix + item for item in analysis.how_to_fix)

    problematic_count = sum(step.status != StepStatus.CORRECT for step in steps)
    if problematic_count:
        summary = f"Проверено шагов: {len(steps)}. Требуют внимания: {problematic_count}."
    else:
        summary = f"Проверено шагов: {len(steps)}. Очевидных ошибок не найдено."

    return AnalysisReport(
        problem=problem,
        steps=steps,
        summary=summary,
        what_is_correct=correct or ["Корректные элементы не выделены автоматически."],
        where_possible_error=errors or ["Явных ошибок не найдено."],
        missing_steps=missing or ["Явных пропущенных шагов не найдено."],
        how_to_fix=fixes or ["Дополнительные исправления не требуются."],
        metadata={
            "step_count": len(steps),
            "contains_latex": contains_latex,
            "problematic_step_count": problematic_count,
        },
    )
