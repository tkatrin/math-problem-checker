from __future__ import annotations

import os
import re
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic import ValidationError

from .schema import Issue, Severity, SolutionStep, StepAnalysis, StepStatus

try:
    from .models.predict import load_default_classifier
except Exception:  # pragma: no cover - ML dependencies can be absent in minimal installs
    load_default_classifier = None


class StepFeedback(BaseModel):
    status: StepStatus
    what_is_correct: list[str] = Field(default_factory=list)
    possible_errors: list[Issue] = Field(default_factory=list)
    missing_steps: list[str] = Field(default_factory=list)
    how_to_fix: list[str] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class StepChecker(Protocol):
    def check_step(
        self,
        *,
        problem: str,
        previous_steps: list[SolutionStep],
        current_step: SolutionStep,
    ) -> StepAnalysis:
        ...


class HeuristicStepChecker:
    """Deterministic fallback that catches obvious MVP-level issues without an API key."""

    def __init__(self, use_ml: bool = True) -> None:
        self.ml_classifier = load_default_classifier() if use_ml and load_default_classifier else None

    def check_step(
        self,
        *,
        problem: str,
        previous_steps: list[SolutionStep],
        current_step: SolutionStep,
    ) -> StepAnalysis:
        text = current_step.text
        issues: list[Issue] = []
        missing_steps: list[str] = []
        fixes: list[str] = []
        correct: list[str] = []

        if any(token in text.lower() for token in ("let", "пусть", "обозначим", "найдем", "найдём")):
            correct.append("Шаг вводит обозначения или явно начинает вычисление.")
        if re.search(r"[=<>≤≥]", text):
            correct.append("Шаг содержит проверяемое математическое утверждение.")

        equalities = re.findall(r"(-?\d+(?:[.,]\d+)?)\s*([+\-*/])\s*(-?\d+(?:[.,]\d+)?)\s*=\s*(-?\d+(?:[.,]\d+)?)(?!\s*/)", text)
        for left, op, right, result in equalities:
            expected = _calculate(left, op, right)
            actual = float(result.replace(",", "."))
            if expected is not None and abs(expected - actual) > 1e-9:
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        title="Возможная вычислительная ошибка",
                        explanation=f"Выражение {left} {op} {right} должно давать {expected:g}, а не {result}.",
                        recommendation="Пересчитайте арифметический переход и исправьте последующие выводы.",
                    )
                )

        probability_issue = _detect_without_replacement_issue(problem, text, previous_steps)
        if probability_issue:
            issues.append(probability_issue)
            missing_steps.append("Нужно явно учесть, что объект выбирается без возвращения и состав множества меняется.")
            fixes.append("Пересчитайте условную вероятность второго выбора с обновлённым числителем и знаменателем.")

        if _has_unjustified_jump(text, previous_steps):
            missing_steps.append("Добавьте промежуточное обоснование перехода к итоговой формуле или ответу.")
            fixes.append("Перед финальным выводом покажите ключевое преобразование, теорему или вычисление.")

        if current_step.index == 1 and not _mentions_problem_objects(problem, text):
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    title="Шаг слабо связан с условием",
                    explanation="В первом шаге не видно явной связи с объектами из условия.",
                    recommendation="Начните с переписывания данных из условия или введения обозначений.",
                )
            )

        status = StepStatus.CORRECT
        if issues:
            status = StepStatus.INCORRECT if any(issue.severity == Severity.ERROR for issue in issues) else StepStatus.NEEDS_ATTENTION
        elif missing_steps:
            status = StepStatus.INCOMPLETE

        if not correct and status == StepStatus.CORRECT:
            correct.append("Явных противоречий на этом шаге не найдено.")
        if issues and not fixes:
            fixes.append("Исправьте отмеченный переход и проверьте, не зависит ли от него следующий шаг.")

        ml_prediction = None
        if self.ml_classifier:
            ml_prediction = self.ml_classifier.predict(
                problem=problem,
                previous_steps=[step.text for step in previous_steps],
                current_step=current_step.text,
                step_index=current_step.index,
            )
            if ml_prediction.label != StepStatus.CORRECT and status == StepStatus.CORRECT:
                status = ml_prediction.label
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        title="ML-модель пометила шаг как подозрительный",
                        explanation=f"Baseline-классификатор предсказал {ml_prediction.label.value} / {ml_prediction.error_type}.",
                        recommendation="Проверьте этот переход вручную или передайте его LLM-объяснителю.",
                    )
                )
                fixes.append("Добавьте пояснение к шагу или перепроверьте формальный переход.")

        return StepAnalysis(
            step=current_step,
            status=status,
            what_is_correct=correct,
            possible_errors=issues,
            missing_steps=missing_steps,
            how_to_fix=fixes,
            confidence=0.45,
            ml_prediction=ml_prediction,
        )


class LLMStepChecker:
    """LangChain/OpenAI checker that asks the model for one structured step analysis."""

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-openai to use LLMStepChecker.") from exc

        selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.llm = ChatOpenAI(model=selected_model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(StepFeedback)

    def check_step(
        self,
        *,
        problem: str,
        previous_steps: list[SolutionStep],
        current_step: SolutionStep,
    ) -> StepAnalysis:
        previous_text = "\n".join(f"{step.index}. {step.text}" for step in previous_steps) or "Нет предыдущих шагов."
        prompt = (
            "Ты проверяешь решение математической задачи по шагам. "
            "Проанализируй только текущий шаг, учитывая условие и предыдущие шаги. "
            "Не решай задачу заново целиком, но укажи логические пробелы, ошибки вычислений и недостающие объяснения.\n\n"
            f"Условие:\n{problem}\n\n"
            f"Предыдущие шаги:\n{previous_text}\n\n"
            f"Текущий шаг {current_step.index}:\n{current_step.text}\n\n"
            "Верни структурированный результат строго по заданной Pydantic-схеме."
        )
        try:
            result = self.structured_llm.invoke(prompt)
            feedback = result if isinstance(result, StepFeedback) else StepFeedback.model_validate(result)
            return StepAnalysis(step=current_step, **feedback.model_dump())
        except (ValidationError, Exception) as exc:
            fallback = HeuristicStepChecker().check_step(
                problem=problem,
                previous_steps=previous_steps,
                current_step=current_step,
            )
            fallback.possible_errors.append(
                Issue(
                    severity=Severity.WARNING,
                    title="LLM-проверка недоступна",
                    explanation=f"Использована локальная эвристика: {exc.__class__.__name__}.",
                    recommendation="Проверьте OPENAI_API_KEY, OPENAI_MODEL и зависимости LangChain/OpenAI.",
                )
            )
            fallback.status = StepStatus.NEEDS_ATTENTION if fallback.status == StepStatus.CORRECT else fallback.status
            return fallback


def build_checker(use_llm: bool = True) -> StepChecker:
    if use_llm and os.getenv("OPENAI_API_KEY"):
        return LLMStepChecker()
    return HeuristicStepChecker()


def _calculate(left: str, op: str, right: str) -> float | None:
    a = float(left.replace(",", "."))
    b = float(right.replace(",", "."))
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/" and b != 0:
        return a / b
    return None


def _has_unjustified_jump(text: str, previous_steps: list[SolutionStep]) -> bool:
    lowered = text.lower()
    final_markers = ("answer", "ответ", "therefore", "следовательно", "итак", "значит")
    has_final_marker = any(marker in lowered for marker in final_markers)
    has_relation = bool(re.search(r"[=<>≤≥]", text)) or any(word in lowered for word in ("равен", "равна", "equals"))
    return has_final_marker and has_relation and len(previous_steps) <= 1 and len(text) < 180


def _mentions_problem_objects(problem: str, text: str) -> bool:
    problem_tokens = {token.lower() for token in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", problem)}
    text_tokens = {token.lower() for token in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", text)}
    return bool(problem_tokens & text_tokens)


def _detect_without_replacement_issue(problem: str, text: str, previous_steps: list[SolutionStep]) -> Issue | None:
    problem_lowered = problem.lower()
    text_lowered = text.lower()
    if "без возвращ" not in problem_lowered and "without replacement" not in problem_lowered:
        return None
    if not any(marker in text_lowered for marker in ("втор", "second")):
        return None

    current_fractions = set(re.findall(r"\b\d+\s*/\s*\d+\b", text))
    previous_fractions = {
        fraction
        for step in previous_steps
        for fraction in re.findall(r"\b\d+\s*/\s*\d+\b", step.text)
    }
    if current_fractions and current_fractions & previous_fractions:
        repeated = sorted(current_fractions & previous_fractions)[0].replace(" ", "")
        return Issue(
            severity=Severity.WARNING,
            title="Возможная ошибка условной вероятности",
            explanation=(
                f"В задаче указан выбор без возвращения, но для второго выбора повторяется вероятность {repeated}. "
                "После первого выбора числитель и знаменатель обычно меняются."
            ),
            recommendation="Используйте условную вероятность для второго события, например пересчитайте количество оставшихся исходов.",
        )
    return None
