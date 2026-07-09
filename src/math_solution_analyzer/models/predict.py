from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from math_solution_analyzer.features import extract_step_features, make_model_text
from math_solution_analyzer.schema import MLStepPrediction, StepStatus


DEFAULT_MODEL_PATH = Path("models/tfidf_logreg.joblib")


class StepMLClassifier:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict(
        self,
        *,
        problem: str,
        previous_steps: list[str],
        current_step: str,
        step_index: int,
    ) -> MLStepPrediction:
        feature_row = extract_step_features(
            problem=problem,
            previous_steps=previous_steps,
            current_step=current_step,
            step_index=step_index,
        )
        feature_row["model_text"] = make_model_text(problem, previous_steps, current_step)
        frame = pd.DataFrame([feature_row])
        label = str(self.pipeline.predict(frame)[0])
        confidence = 0.0
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(frame)[0]
            confidence = float(max(probabilities))
        return MLStepPrediction(
            label=StepStatus(label),
            error_type=_guess_error_type(problem, current_step, label),
            confidence=round(confidence, 4),
            model_name=self.model_path.name,
        )


def load_default_classifier() -> StepMLClassifier | None:
    if not DEFAULT_MODEL_PATH.exists():
        return None
    try:
        return StepMLClassifier(DEFAULT_MODEL_PATH)
    except Exception:
        return None


def _guess_error_type(problem: str, current_step: str, label: str) -> str:
    text = f"{problem} {current_step}".lower()
    if label == "correct":
        return "none"
    if "без возвращ" in text and any(marker in text for marker in ("втор", "second")):
        return "probability_without_replacement"
    if any(marker in text for marker in ("производная", "derivative", "sin", "cos")):
        return "wrong_formula"
    if any(marker in text for marker in ("ответ", "answer")):
        return "wrong_final_answer"
    if any(op in current_step for op in ("+", "-", "*", "/")) and "=" in current_step:
        return "calculation_error"
    return "invalid_transition"
