from math_solution_analyzer import analyze_solution
from math_solution_analyzer.schema import StepAnalysis
from math_solution_analyzer.parser import parse_input
from math_solution_analyzer.step_splitter import split_solution_into_steps


class FakeExplanationGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def explain_step(self, *, problem: str, analysis: StepAnalysis) -> str:
        self.calls += 1
        return f"Пояснение для шага {analysis.step.index}"


def test_parse_detects_latex() -> None:
    parsed = parse_input("Найдите $x^2$.", "Ответ: $2x$.")
    assert parsed.contains_latex is True


def test_step_splitter_uses_numbered_steps() -> None:
    steps = split_solution_into_steps("1. Пусть x=1.\n2. Тогда x+1=2.")
    assert [step.index for step in steps] == [1, 2]
    assert steps[0].text == "Пусть x=1."


def test_step_splitter_keeps_single_numbered_step() -> None:
    steps = split_solution_into_steps("1. Найдите сумму чисел: 2 + 2 = 4.")
    assert len(steps) == 1
    assert steps[0].text == "Найдите сумму чисел: 2 + 2 = 4."


def test_pipeline_catches_arithmetic_error_without_llm() -> None:
    report = analyze_solution(
        "Вычислите 2 + 2.",
        "1. Складываем числа.\n2. 2 + 2 = 5.\n3. Ответ: 5.",
        use_llm=False,
    )
    assert report.metadata["step_count"] == 3
    assert any("должно давать 4" in item for item in report.where_possible_error)
    assert report.steps[2].status.value == "needs_attention"
    assert any("зависит от предыдущей ошибки" in issue.title.lower() for issue in report.steps[2].possible_errors)


def test_pipeline_returns_required_sections() -> None:
    report = analyze_solution(
        "Найдите производную f(x)=x^2.",
        "1. Используем правило степени.\n2. f'(x)=2x.",
        use_llm=False,
    )
    assert report.what_is_correct
    assert report.where_possible_error
    assert report.missing_steps
    assert report.how_to_fix


def test_pipeline_uses_llm_explanation_generator_when_enabled() -> None:
    generator = FakeExplanationGenerator()
    report = analyze_solution(
        "Вычислите 2 + 2.",
        "1. Складываем числа.\n2. 2 + 2 = 5.",
        use_llm=True,
        explanation_generator=generator,
    )
    assert generator.calls == 2
    assert report.steps[0].llm_explanation == "Пояснение для шага 1"


def test_pipeline_skips_llm_explanation_generator_when_disabled() -> None:
    generator = FakeExplanationGenerator()
    report = analyze_solution(
        "Вычислите 2 + 2.",
        "1. Складываем числа.",
        use_llm=False,
        explanation_generator=generator,
    )
    assert generator.calls == 0
    assert report.steps[0].llm_explanation is None


def test_pipeline_skips_llm_for_correct_steps() -> None:
    generator = FakeExplanationGenerator()
    report = analyze_solution(
        "Найдите сумму чисел 2 и 2.",
        "1. Найдите сумму чисел: 2 + 2 = 4.",
        use_llm=True,
        use_ml=False,
        explanation_generator=generator,
    )
    assert generator.calls == 0
    assert all(step.llm_explanation is None for step in report.steps)
