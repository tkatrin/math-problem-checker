from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

from math_solution_analyzer.models.train import _build_features_frame


def evaluate(model_path: Path, dataset_path: Path, metrics_path: Path, confusion_matrix_path: Path, seed: int = 42) -> dict:
    df = pd.read_csv(dataset_path)
    X = _build_features_frame(df)
    y_label = df["label"].astype(str)
    y_error_type = df["error_type"].astype(str)
    groups = df["problem_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    _, test_idx = next(splitter.split(X, y_label, groups=groups))
    X_test = X.iloc[test_idx]
    y_label_test = y_label.iloc[test_idx]
    y_error_test = y_error_type.iloc[test_idx]
    df_test = df.iloc[test_idx]

    artifact = joblib.load(model_path)
    label_model = artifact["label_model"] if isinstance(artifact, dict) else artifact
    error_type_model = artifact.get("error_type_model") if isinstance(artifact, dict) else None
    label_predictions = label_model.predict(X_test)
    error_predictions = error_type_model.predict(X_test) if error_type_model is not None else ["none"] * len(X_test)
    rule_predictions = [_rule_label_prediction(row) for _, row in df_test.iterrows()]

    metrics = {
        "split": "GroupShuffleSplit by problem_id",
        "test_rows": int(len(test_idx)),
        "test_groups": int(groups.iloc[test_idx].nunique()),
        "rule_based": {
            "label_accuracy": round(float(accuracy_score(y_label_test, rule_predictions)), 4),
            "label_macro_f1": round(float(f1_score(y_label_test, rule_predictions, average="macro")), 4),
            "step_level_f1": round(float(f1_score(list(y_label_test != "correct"), [label != "correct" for label in rule_predictions])), 4),
        },
        "tfidf_logreg": {
            "label_accuracy": round(float(accuracy_score(y_label_test, label_predictions)), 4),
            "label_macro_f1": round(float(f1_score(y_label_test, label_predictions, average="macro")), 4),
            "step_level_f1": round(float(f1_score(list(y_label_test != "correct"), [label != "correct" for label in label_predictions])), 4),
            "error_type_accuracy": round(float(accuracy_score(y_error_test, error_predictions)), 4),
            "error_type_macro_f1": round(float(f1_score(y_error_test, error_predictions, average="macro")), 4),
            "label_report": classification_report(y_label_test, label_predictions, output_dict=True, zero_division=0),
            "error_type_report": classification_report(y_error_test, error_predictions, output_dict=True, zero_division=0),
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_label_test, label_predictions, xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=180)
    plt.close()
    return metrics


def _rule_label_prediction(row: pd.Series) -> str:
    text = str(row["current_step"]).lower()
    problem = str(row["problem"]).lower()
    previous = str(row.get("previous_steps", ""))
    if _has_arithmetic_error(str(row["current_step"])):
        return "incorrect"
    if "без возвращ" in problem and "втор" in text:
        fractions = set(__import__("re").findall(r"\b\d+\s*/\s*\d+\b", str(row["current_step"])))
        previous_fractions = set(__import__("re").findall(r"\b\d+\s*/\s*\d+\b", previous))
        if fractions & previous_fractions:
            return "incorrect"
    if any(marker in text for marker in ("очевидно", "сразу", "без дополнительных", "аналогична", "известный факт")):
        return "suspicious"
    if any(marker in text for marker in ("следовательно", "значит", "ответ")) and "=" not in text and "/" not in text:
        return "incomplete"
    return "correct"


def _has_arithmetic_error(text: str) -> bool:
    import re

    for left, op, right, result in re.findall(r"(-?\d+(?:[.,]\d+)?)\s*([+\-*/])\s*(-?\d+(?:[.,]\d+)?)\s*=\s*(-?\d+(?:[.,]\d+)?)(?!\s*/)", text):
        a = float(left.replace(",", "."))
        b = float(right.replace(",", "."))
        actual = float(result.replace(",", "."))
        expected = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else actual}.get(op, actual)
        if abs(expected - actual) > 1e-9:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/tfidf_logreg.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/step_classification.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--confusion-matrix", type=Path, default=Path("reports/confusion_matrix.png"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.dataset, args.metrics, args.confusion_matrix), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
