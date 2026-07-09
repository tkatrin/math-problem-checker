from pathlib import Path
import json
import subprocess
import sys

import pandas as pd

from math_solution_analyzer.dataset import ERROR_TYPES, build_synthetic_dataset, save_dataset
from math_solution_analyzer.evaluation import evaluate
from math_solution_analyzer.features import extract_step_features
from math_solution_analyzer.models.predict import StepMLClassifier
from math_solution_analyzer.models.train import train_baseline
from math_solution_analyzer.schema import MLStepPrediction


def test_dataset_has_required_columns_and_classes(tmp_path: Path) -> None:
    rows = build_synthetic_dataset(n=500, seed=7)
    output = tmp_path / "dataset.csv"
    save_dataset(rows, output)
    df = pd.read_csv(output)

    expected_columns = {
        "problem_id",
        "domain",
        "problem",
        "previous_steps",
        "current_step",
        "next_step",
        "step_index",
        "label",
        "error_type",
        "explanation",
    }
    assert expected_columns.issubset(df.columns)
    assert {"correct", "incorrect", "incomplete", "suspicious"}.issubset(set(df["label"]))
    assert set(ERROR_TYPES).issubset(set(df["error_type"]))


def test_feature_extractor_returns_stable_features() -> None:
    features = extract_step_features(
        problem="Вычислите 2 + 2.",
        previous_steps=["Складываем числа."],
        current_step="2 + 2 = 5.",
        step_index=2,
    )
    assert "sympy_error" in features
    assert "domain_arithmetic" in features
    assert "number_count" in features
    assert features["sympy_error"] == 1


def test_train_predict_and_evaluate_save_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "step_classification.csv"
    model_path = tmp_path / "tfidf_logreg.joblib"
    metrics_path = tmp_path / "metrics.json"
    evaluation_path = tmp_path / "evaluation.json"
    confusion_path = tmp_path / "confusion_matrix.png"

    save_dataset(build_synthetic_dataset(n=500, seed=11), dataset_path)
    metrics = train_baseline(dataset_path, model_path, metrics_path)
    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["split"] == "GroupShuffleSplit by problem_id"
    assert "error_type_macro_f1" in metrics

    classifier = StepMLClassifier(model_path)
    prediction = classifier.predict(
        problem="Вычислите 2 + 2.",
        previous_steps=["Складываем числа."],
        current_step="2 + 2 = 5.",
        step_index=2,
    )
    assert isinstance(prediction, MLStepPrediction)
    assert prediction.error_type

    evaluation = evaluate(model_path, dataset_path, evaluation_path, confusion_path)
    assert evaluation_path.exists()
    assert confusion_path.exists()
    assert "rule_based" in evaluation
    assert "tfidf_logreg" in evaluation


def test_cli_check_outputs_json(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.txt"
    solution_path = tmp_path / "solution.txt"
    problem_path.write_text("Вычислите 2 + 2.", encoding="utf-8")
    solution_path.write_text("1. Складываем числа.\n2. 2 + 2 = 5.", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "math_solution_analyzer.cli",
            "check",
            "--problem",
            str(problem_path),
            "--solution",
            str(solution_path),
            "--no-llm",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["metadata"]["step_count"] == 2
    assert payload["steps"][1]["ml_prediction"]["error_type"]
