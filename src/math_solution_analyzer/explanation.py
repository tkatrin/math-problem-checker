from __future__ import annotations

import os
from typing import Protocol

from .schema import StepAnalysis


class ExplanationGenerator(Protocol):
    def explain_step(self, *, problem: str, analysis: StepAnalysis) -> str:
        ...


class LLMExplanationGenerator:
    """Turns rule/ML signals into a short human-readable explanation."""

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-openai to use LLMExplanationGenerator.") from exc

        selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.llm = ChatOpenAI(model=selected_model, temperature=temperature)

    def explain_step(self, *, problem: str, analysis: StepAnalysis) -> str:
        ml = analysis.ml_prediction
        issues = "\n".join(f"- {issue.title}: {issue.explanation}" for issue in analysis.possible_errors) or "- Нет явных замечаний."
        missing = "\n".join(f"- {item}" for item in analysis.missing_steps) or "- Нет явных пропусков."
        fixes = "\n".join(f"- {item}" for item in analysis.how_to_fix) or "- Исправления не требуются."
        ml_text = (
            f"label={ml.label.value}, error_type={ml.error_type}, confidence={ml.confidence:.3f}"
            if ml
            else "ML prediction unavailable"
        )
        prompt = (
            "Ты формулируешь объяснение для результата ML/правил проверки математического решения. "
            "Не решай задачу заново и не добавляй новые выводы, которых нет в сигналах. "
            "Сделай короткое объяснение на русском: что система считает проблемой и как исправить.\n\n"
            f"Условие:\n{problem}\n\n"
            f"Шаг {analysis.step.index}:\n{analysis.step.text}\n\n"
            f"Статус: {analysis.status.value}\n"
            f"ML: {ml_text}\n\n"
            f"Замечания:\n{issues}\n\n"
            f"Пропущенные шаги:\n{missing}\n\n"
            f"Рекомендации:\n{fixes}\n"
        )
        response = self.llm.invoke(prompt)
        return str(getattr(response, "content", response)).strip()


def build_explanation_generator(use_llm: bool = True) -> ExplanationGenerator | None:
    if not use_llm or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return LLMExplanationGenerator()
    except Exception:
        return None
