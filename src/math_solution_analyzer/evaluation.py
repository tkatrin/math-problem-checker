from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from math_solution_analyzer.models.train import _build_features_frame


def evaluate(model_path: Path, dataset_path: Path, metrics_path: Path, confusion_matrix_path: Path, seed: int = 42) -> dict:
    df = pd.read_csv(dataset_path)
    X = _build_features_frame(df)
    y = df["label"].astype(str)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "step_level_f1": round(float(f1_score(y_test != "correct", predictions != "correct")), 4),
        "macro_f1_by_label": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=180)
    plt.close()
    return metrics


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
