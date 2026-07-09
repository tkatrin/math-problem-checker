from math_solution_analyzer import analyze_solution
from math_solution_analyzer.parser import parse_input
from math_solution_analyzer.step_splitter import split_solution_into_steps


def test_parse_detects_latex() -> None:
    parsed = parse_input("Найдите $x^2$.", "Ответ: $2x$.")
    assert parsed.contains_latex is True


def test_step_splitter_uses_numbered_steps() -> None:
    steps = split_solution_into_steps("1. Пусть x=1.\n2. Тогда x+1=2.")
    assert [step.index for step in steps] == [1, 2]
    assert steps[0].text == "Пусть x=1."


def test_pipeline_catches_arithmetic_error_without_llm() -> None:
    report = analyze_solution(
        "Вычислите 2 + 2.",
        "1. Складываем числа.\n2. 2 + 2 = 5.\n3. Ответ: 5.",
        use_llm=False,
    )
    assert report.metadata["step_count"] == 3
    assert any("должно давать 4" in item for item in report.where_possible_error)


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
