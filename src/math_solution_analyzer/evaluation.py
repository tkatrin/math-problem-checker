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


def evaluate(
    model_path: Path,
    dataset_path: Path,
    metrics_path: Path,
    confusion_matrix_path: Path,
    seed: int = 42,
    *,
    split_eval: bool = True,
    threshold: float = 0.5,
    first_error_strategy: str = "hybrid",
    predictions_path: Path | None = None,
) -> dict:
    _validate_first_error_strategy(first_error_strategy)
    df = pd.read_csv(dataset_path)
    X = _build_features_frame(df)
    y_label = df["label"].astype(str)
    y_error_type = df["error_type"].astype(str)
    groups = df["problem_id"].astype(str)
    sources = df["source"].astype(str) if "source" in df.columns else pd.Series(["unknown"] * len(df))
    if split_eval:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        _, test_idx = next(splitter.split(X, y_label, groups=groups))
        split_name = "GroupShuffleSplit by problem_id"
    else:
        test_idx = list(range(len(df)))
        split_name = "external eval dataset"

    X_test = X.iloc[test_idx]
    y_label_test = y_label.iloc[test_idx]
    y_error_test = y_error_type.iloc[test_idx]
    df_test = df.iloc[test_idx].copy()
    source_test = sources.iloc[test_idx]

    artifact = joblib.load(model_path)
    label_model = artifact["label_model"] if isinstance(artifact, dict) else artifact
    error_type_model = artifact.get("error_type_model") if isinstance(artifact, dict) else None
    label_predictions = label_model.predict(X_test)
    error_predictions = error_type_model.predict(X_test) if error_type_model is not None else ["none"] * len(X_test)
    label_probabilities = _predict_label_probabilities(label_model, X_test)
    rule_predictions = [_rule_label_prediction(row) for _, row in df_test.iterrows()]
    prediction_rows = _build_prediction_rows(df_test, label_predictions, error_predictions, label_probabilities)
    first_error_metrics = _first_error_metrics(df_test, prediction_rows, threshold=threshold, strategy=first_error_strategy)

    metrics = {
        "split": split_name,
        "test_rows": int(len(test_idx)),
        "test_groups": int(groups.iloc[test_idx].nunique()),
        "eval_sources": _source_counts(source_test),
        "metrics_by_source": _metrics_by_source(df_test, prediction_rows, threshold=threshold, strategy=first_error_strategy),
        "rule_based": {
            "label_accuracy": round(float(accuracy_score(y_label_test, rule_predictions)), 4),
            "label_macro_f1": round(float(f1_score(y_label_test, rule_predictions, average="macro")), 4),
            "step_level_f1": round(float(f1_score(list(y_label_test != "correct"), [label != "correct" for label in rule_predictions])), 4),
        },
        "tfidf_logreg": {
            "label_accuracy": round(float(accuracy_score(y_label_test, label_predictions)), 4),
            "label_macro_f1": round(float(f1_score(y_label_test, label_predictions, average="macro")), 4),
            "step_level_f1": round(float(f1_score(list(y_label_test != "correct"), [label != "correct" for label in label_predictions])), 4),
            "first_error_accuracy": first_error_metrics["first_error_accuracy"],
            "first_error_macro_f1": first_error_metrics["first_error_macro_f1"],
            "all_correct_accuracy": first_error_metrics["all_correct_accuracy"],
            "first_error_threshold": threshold,
            "first_error_strategy": first_error_strategy,
            "error_type_accuracy": round(float(accuracy_score(y_error_test, error_predictions)), 4),
            "error_type_macro_f1": round(float(f1_score(y_error_test, error_predictions, average="macro")), 4),
            "label_report": classification_report(y_label_test, label_predictions, output_dict=True, zero_division=0),
            "error_type_report": classification_report(y_error_test, error_predictions, output_dict=True, zero_division=0),
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(json.dumps(prediction_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_label_test, label_predictions, xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=180)
    plt.close()
    return metrics


def _predict_label_probabilities(label_model, X_test: pd.DataFrame) -> list[dict[str, float]]:
    if not hasattr(label_model, "predict_proba"):
        return [{} for _ in range(len(X_test))]
    probabilities = label_model.predict_proba(X_test)
    classes = [str(item) for item in label_model.classes_]
    return [
        {class_name: round(float(probability), 6) for class_name, probability in zip(classes, row)}
        for row in probabilities
    ]


def _build_prediction_rows(
    df_test: pd.DataFrame,
    label_predictions,
    error_predictions,
    label_probabilities: list[dict[str, float]],
) -> list[dict]:
    rows: list[dict] = []
    for (_, row), label, error_type, probabilities in zip(df_test.iterrows(), label_predictions, error_predictions, label_probabilities):
        rows.append(
            {
                "problem_id": str(row["problem_id"]),
                "source": str(row.get("source", "unknown")),
                "step_index": int(row["step_index"]),
                "problem": str(row["problem"]),
                "current_step": str(row["current_step"]),
                "true_label": str(row["label"]),
                "predicted_label": str(label),
                "true_error_type": str(row["error_type"]),
                "predicted_error_type": str(error_type),
                "label_probabilities": probabilities,
                "p_correct": float(probabilities.get("correct", 0.0)),
                "p_incorrect": float(probabilities.get("incorrect", 0.0)),
                "p_suspicious": float(probabilities.get("suspicious", 0.0)),
            }
        )
    return rows


def _first_error_metrics(df_test: pd.DataFrame, prediction_rows: list[dict], *, threshold: float, strategy: str) -> dict[str, float | str | None]:
    true_by_problem = _first_error_by_problem(
        [
            {
                "problem_id": str(row["problem_id"]),
                "step_index": int(row["step_index"]),
                "label": str(row["label"]),
            }
            for _, row in df_test.iterrows()
        ],
        label_key="label",
    )
    pred_by_problem = _predicted_first_error_by_problem(prediction_rows, threshold=threshold, strategy=strategy)
    problem_ids = sorted(true_by_problem)
    y_true = [true_by_problem[problem_id] for problem_id in problem_ids]
    y_pred = [pred_by_problem.get(problem_id, -1) for problem_id in problem_ids]
    all_correct_positions = [index for index, value in enumerate(y_true) if value == -1]
    all_correct_accuracy: float | str = "not_applicable"
    if all_correct_positions:
        all_correct_accuracy = round(
            sum(1 for index in all_correct_positions if y_pred[index] == -1) / len(all_correct_positions),
            4,
        )
    return {
        "first_error_accuracy": round(float(accuracy_score(y_true, y_pred)), 4) if y_true else None,
        "first_error_macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4) if y_true else None,
        "all_correct_accuracy": all_correct_accuracy,
    }


def _first_error_by_problem(rows: list[dict], *, label_key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in sorted(rows, key=lambda item: (item["problem_id"], item["step_index"])):
        problem_id = row["problem_id"]
        result.setdefault(problem_id, -1)
        if result[problem_id] == -1 and row[label_key] == "incorrect":
            result[problem_id] = int(row["step_index"])
    return result


def _predicted_first_error_by_problem(prediction_rows: list[dict], *, threshold: float, strategy: str) -> dict[str, int]:
    _validate_first_error_strategy(strategy)
    result: dict[str, int] = {}
    for row in sorted(prediction_rows, key=lambda item: (item["problem_id"], item["step_index"])):
        problem_id = row["problem_id"]
        result.setdefault(problem_id, -1)
        if result[problem_id] != -1:
            continue
        if _is_predicted_error_step(row, threshold=threshold, strategy=strategy):
            result[problem_id] = int(row["step_index"])
    return result


def _is_predicted_error_step(row: dict, *, threshold: float, strategy: str) -> bool:
    if strategy == "threshold":
        return row["p_incorrect"] >= threshold
    if strategy == "hard_label":
        return row["predicted_label"] == "incorrect"
    return row["p_incorrect"] >= threshold or row["predicted_label"] == "incorrect"


def _validate_first_error_strategy(strategy: str) -> None:
    if strategy not in {"threshold", "hard_label", "hybrid"}:
        raise ValueError("first_error_strategy must be one of: threshold, hard_label, hybrid")


def _metrics_by_source(df_test: pd.DataFrame, prediction_rows: list[dict], *, threshold: float, strategy: str) -> dict[str, dict]:
    predictions = pd.DataFrame(prediction_rows)
    metrics: dict[str, dict] = {}
    for source, source_df in df_test.groupby(df_test["source"] if "source" in df_test.columns else pd.Series(["unknown"] * len(df_test), index=df_test.index)):
        source_predictions = predictions[predictions["source"] == str(source)]
        source_first_error = _first_error_metrics(source_df, source_predictions.to_dict("records"), threshold=threshold, strategy=strategy)
        metrics[str(source)] = {
            "rows": int(len(source_df)),
            "problems": int(source_df["problem_id"].nunique()),
            "label_macro_f1": round(
                float(f1_score(source_df["label"].astype(str), source_predictions["predicted_label"].astype(str), average="macro")),
                4,
            ),
            "first_error_accuracy": source_first_error["first_error_accuracy"],
            "first_error_macro_f1": source_first_error["first_error_macro_f1"],
            "all_correct_accuracy": source_first_error["all_correct_accuracy"],
        }
    return metrics


def _source_counts(sources: pd.Series) -> dict[str, int]:
    return {str(source): int(count) for source, count in sources.value_counts().sort_index().items()}


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
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset CSV path. Uses group holdout by default.")
    parser.add_argument("--eval-dataset", type=Path, default=None, help="External eval dataset CSV path. Evaluates all rows.")
    parser.add_argument("--metrics", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--confusion-matrix", type=Path, default=Path("reports/confusion_matrix.png"))
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--first-error-threshold", type=float, default=0.5)
    parser.add_argument("--first-error-strategy", choices=["threshold", "hard_label", "hybrid"], default="hybrid")
    args = parser.parse_args()
    dataset_path = args.eval_dataset or args.dataset or Path("data/processed/step_classification.csv")
    print(
        json.dumps(
            evaluate(
                args.model,
                dataset_path,
                args.metrics,
                args.confusion_matrix,
                split_eval=args.eval_dataset is None,
                threshold=args.first_error_threshold,
                first_error_strategy=args.first_error_strategy,
                predictions_path=args.predictions,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
