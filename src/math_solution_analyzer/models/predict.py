from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from math_solution_analyzer.features import extract_step_features, make_model_text
from math_solution_analyzer.models.binary_benchmark import apply_symbolic_overrides, make_binary_model_text
from math_solution_analyzer.schema import MLStepPrediction, StepStatus


DEFAULT_MODEL_PATH = Path("models/tfidf_logreg.joblib")
IMPROVED_MODEL_PATH = Path("models/binary_prm800k_target_adapted.joblib")


class StepMLClassifier:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        artifact = joblib.load(model_path)
        self.artifact_type = artifact.get("artifact_type") if isinstance(artifact, dict) else None
        self.step_threshold = float(artifact.get("step_threshold", 0.5)) if isinstance(artifact, dict) else 0.5
        if isinstance(artifact, dict):
            self.label_model = artifact["label_model"]
            self.error_type_model = artifact.get("error_type_model")
        else:
            self.label_model = artifact
            self.error_type_model = None

    def predict(
        self,
        *,
        problem: str,
        previous_steps: list[str],
        current_step: str,
        step_index: int,
    ) -> MLStepPrediction:
        binary_mode = self.artifact_type == "binary_step_classifier"
        feature_row = extract_step_features(
            problem=problem,
            previous_steps=previous_steps,
            current_step=current_step,
            step_index=step_index,
            expensive_symbolic=not binary_mode,
        )
        feature_row["model_text"] = (
            make_binary_model_text(problem, " ||| ".join(previous_steps), current_step)
            if binary_mode
            else make_model_text(problem, previous_steps, current_step)
        )
        frame = pd.DataFrame([feature_row])
        label = str(self.label_model.predict(frame)[0])
        confidence = 0.0
        label_probabilities: dict[str, float] = {}
        if hasattr(self.label_model, "predict_proba"):
            probabilities = self.label_model.predict_proba(frame)[0]
            if binary_mode:
                classes = list(self.label_model.classes_)
                p_incorrect = float(probabilities[classes.index(1)])
                p_incorrect = float(apply_symbolic_overrides(frame, np.array([p_incorrect]))[0])
                label = "incorrect" if p_incorrect >= self.step_threshold else "correct"
                confidence = p_incorrect if label == "incorrect" else 1.0 - p_incorrect
                label_probabilities = {
                    "correct": round(1.0 - p_incorrect, 4),
                    "incorrect": round(p_incorrect, 4),
                }
            else:
                confidence = float(max(probabilities))
                label_probabilities = {
                    str(label_name): round(float(probability), 4)
                    for label_name, probability in zip(self.label_model.classes_, probabilities)
                }
        error_type = "none"
        if self.error_type_model is not None:
            error_type = str(self.error_type_model.predict(frame)[0])
        else:
            error_type = _guess_error_type(problem, current_step, label)
        return MLStepPrediction(
            label=StepStatus(label),
            error_type=error_type,
            confidence=round(confidence, 4),
            model_name=self.model_path.name,
            label_probabilities=label_probabilities,
        )


def load_default_classifier() -> StepMLClassifier | None:
    for model_path in (IMPROVED_MODEL_PATH, DEFAULT_MODEL_PATH):
        if not model_path.exists():
            continue
        try:
            return StepMLClassifier(model_path)
        except Exception:
            continue
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
