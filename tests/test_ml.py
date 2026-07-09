from pathlib import Path
import json
import subprocess
import sys

import pandas as pd

from math_solution_analyzer.dataset import ERROR_TYPES, build_synthetic_dataset, save_dataset
from math_solution_analyzer.data_sources.prm800k import normalize_prm800k_record
from math_solution_analyzer.data_sources.processbench import normalize_processbench_record
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
        "source",
    }
    assert expected_columns.issubset(df.columns)
    assert {"correct", "incorrect", "incomplete", "suspicious"}.issubset(set(df["label"]))
    assert set(ERROR_TYPES).issubset(set(df["error_type"]))
    assert set(df["source"]) == {"toy_synthetic_baseline"}


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


def test_cli_supports_output_and_no_ml(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.txt"
    solution_path = tmp_path / "solution.txt"
    output_path = tmp_path / "report.json"
    problem_path.write_text("Вычислите 2 + 2.", encoding="utf-8")
    solution_path.write_text("1. Складываем числа.\n2. 2 + 2 = 5.", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "math_solution_analyzer.cli",
            "check",
            "--problem",
            str(problem_path),
            "--solution",
            str(solution_path),
            "--output",
            str(output_path),
            "--no-ml",
            "--no-llm",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["ml_enabled"] is False
    assert payload["steps"][1]["ml_prediction"] is None


def test_prm800k_adapter_normalizes_step_labels() -> None:
    record = {
        "timestamp": "sample-1",
        "question": {"problem": "Compute 2+2."},
        "label": {
            "steps": [
                {"chosen_completion": 0, "human_completion": None, "completions": [{"text": "2+2=4", "rating": 1}]},
                {"chosen_completion": 0, "human_completion": None, "completions": [{"text": "4 is final", "rating": 0}]},
                {"chosen_completion": 0, "human_completion": None, "completions": [{"text": "2+2=5", "rating": -1}]},
            ]
        },
    }
    rows = normalize_prm800k_record(record)
    assert [row["label"] for row in rows] == ["correct", "suspicious", "incorrect"]
    assert rows[2]["error_type"] == "process_error"
    assert rows[2]["source"] == "prm800k"


def test_processbench_adapter_marks_first_error() -> None:
    record = {
        "id": "case-1",
        "problem": "Compute 2+2.",
        "steps": ["Set up addition.", "2+2=5.", "Answer is 5."],
        "label": 1,
    }
    rows = normalize_processbench_record(record)
    assert [row["label"] for row in rows] == ["correct", "incorrect", "suspicious"]
    assert rows[1]["error_type"] == "process_error"
    assert rows[2]["error_type"] == "after_error_context"
