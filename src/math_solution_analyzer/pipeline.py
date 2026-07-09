from __future__ import annotations

from pathlib import Path

from .checker import RuleBasedChecker, StepChecker
from .explanation import ExplanationGenerator, build_explanation_generator
from .models.predict import StepMLClassifier, load_default_classifier
from .parser import parse_input
from .report_generator import generate_report
from .schema import AnalysisReport, Issue, MLStepPrediction, Severity, StepAnalysis, StepStatus
from .step_splitter import split_solution_into_steps


def analyze_solution(
    problem: str,
    solution: str,
    *,
    use_llm: bool = True,
    checker: StepChecker | None = None,
    override_checker: StepChecker | None = None,
    ml_classifier: StepMLClassifier | None = None,
    model_path: Path | None = None,
    use_ml: bool = True,
    explanation_generator: ExplanationGenerator | None = None,
) -> AnalysisReport:
    """Analyze a solution.

    override_checker bypasses the default rule/ML pipeline and is intended for tests or
    custom integrations. The legacy checker argument is kept as an alias.
    """

    parsed = parse_input(problem, solution)
    steps = split_solution_into_steps(parsed.solution)
    metadata: dict[str, object] = {}

    analyses: list[StepAnalysis] = []
    selected_override_checker = override_checker or checker
    if selected_override_checker is not None:
        for step in steps:
            previous_steps = [analysis.step for analysis in analyses]
            analyses.append(
                selected_override_checker.check_step(
                    problem=parsed.problem,
                    previous_steps=previous_steps,
                    current_step=step,
                )
            )
        report = generate_report(parsed.problem, analyses, contains_latex=parsed.contains_latex)
        report.metadata["checker_override"] = True
        return report

    rule_checker = RuleBasedChecker()
    selected_ml_classifier = None
    if use_ml:
        if ml_classifier is not None:
            selected_ml_classifier = ml_classifier
        elif model_path is not None:
            selected_ml_classifier = StepMLClassifier(model_path)
        else:
            selected_ml_classifier = load_default_classifier()
    selected_explanation_generator = explanation_generator if use_llm else None
    if selected_explanation_generator is None:
        selected_explanation_generator = build_explanation_generator(use_llm=use_llm)
    if use_llm and selected_explanation_generator is None:
        metadata["llm_warning"] = "OpenAI explanation unavailable, rule/ML analysis completed."

    for step in steps:
        previous_steps = [analysis.step for analysis in analyses]
        analysis = rule_checker.check_step(
            problem=parsed.problem,
            previous_steps=previous_steps,
            current_step=step,
        )
        if selected_ml_classifier is not None:
            ml_prediction = selected_ml_classifier.predict(
                problem=parsed.problem,
                previous_steps=[previous_step.text for previous_step in previous_steps],
                current_step=step.text,
                step_index=step.index,
            )
            _merge_ml_prediction(analysis, ml_prediction)
        if selected_explanation_generator is not None and analysis.status != StepStatus.CORRECT:
            _attach_llm_explanation(selected_explanation_generator, parsed.problem, analysis)
        analyses.append(analysis)

    report = generate_report(parsed.problem, analyses, contains_latex=parsed.contains_latex)
    report.metadata.update(metadata)
    report.metadata["ml_enabled"] = selected_ml_classifier is not None
    report.metadata["llm_enabled"] = selected_explanation_generator is not None
    return report


def _merge_ml_prediction(analysis: StepAnalysis, prediction: MLStepPrediction) -> None:
    analysis.ml_prediction = prediction
    if prediction.label == StepStatus.CORRECT:
        return

    if analysis.status == StepStatus.CORRECT or _ml_label_is_more_severe(prediction.label, analysis.status):
        analysis.status = prediction.label

    analysis.possible_errors.append(
        Issue(
            severity=Severity.WARNING if prediction.label != StepStatus.INCORRECT else Severity.ERROR,
            title="ML-классификатор отметил шаг",
            explanation=(
                f"Модель предсказала label={prediction.label.value}, "
                f"error_type={prediction.error_type}, confidence={prediction.confidence:.3f}."
            ),
            recommendation="Сравните ML-сигнал с rule-based проверкой и добавьте недостающее обоснование или исправьте переход.",
        )
    )
    if prediction.label == StepStatus.INCOMPLETE:
        analysis.missing_steps.append("ML-модель считает, что в этом месте может не хватать промежуточного шага.")
    if prediction.label == StepStatus.SUSPICIOUS:
        analysis.how_to_fix.append("Уточните рассуждение: покажите формулу, подстановку или используемое правило.")


def _attach_llm_explanation(generator: ExplanationGenerator, problem: str, analysis: StepAnalysis) -> None:
    try:
        analysis.llm_explanation = generator.explain_step(problem=problem, analysis=analysis)
    except Exception as exc:
        analysis.possible_errors.append(
            Issue(
                severity=Severity.WARNING,
                title="LLM-объяснение недоступно",
                explanation=f"Не удалось сформировать LLM-объяснение: {exc.__class__.__name__}.",
                recommendation="Проверьте OPENAI_API_KEY, OPENAI_MODEL и сетевой доступ. Rule/ML-анализ уже выполнен.",
            )
        )


def _ml_label_is_more_severe(candidate: StepStatus, current: StepStatus) -> bool:
    order = {
        StepStatus.CORRECT: 0,
        StepStatus.SUSPICIOUS: 1,
        StepStatus.NEEDS_ATTENTION: 1,
        StepStatus.INCOMPLETE: 2,
        StepStatus.INCORRECT: 3,
    }
    return order[candidate] > order[current]
