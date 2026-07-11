from pathlib import Path
import json
import subprocess
import sys

import pandas as pd

from math_solution_analyzer.dataset import ERROR_TYPES, build_synthetic_dataset, save_dataset
from math_solution_analyzer.data_sources.common import iter_json_records
from math_solution_analyzer.data_sources.prm800k import normalize_prm800k_record
from math_solution_analyzer.data_sources.processbench import normalize_processbench_record
from math_solution_analyzer.data_sources.common import write_rows_csv
from math_solution_analyzer.evaluation import evaluate
from math_solution_analyzer.experiments.prm800k_to_processbench import _format_examples, run_experiment
from math_solution_analyzer.experiments.binary_prm800k_scaling import split_target_domain_groups
from math_solution_analyzer.features import (
    check_division_remainder,
    check_polynomial_factorization,
    extract_step_features,
    sympy_arithmetic_error,
)
from math_solution_analyzer.models.binary_benchmark import (
    first_error_metrics,
    make_binary_model_text,
    select_problem_groups_by_step_budget,
)
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
    assert metrics["train_sources"]["toy_synthetic_baseline"] == 500
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
    assert "metrics_by_source" in evaluation


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


def test_json_records_reads_json_array_export(tmp_path: Path) -> None:
    source = tmp_path / "processbench.json"
    source.write_text('[{"id": "case-1"}]', encoding="utf-8")
    assert list(iter_json_records(source)) == [{"id": "case-1"}]


def test_error_analysis_formats_step_text() -> None:
    lines = _format_examples(
        [
            {
                "problem_id": "case-1",
                "step_index": 1,
                "true_label": "correct",
                "predicted_label": "incorrect",
                "p_incorrect": 0.8,
                "current_step": "Set up addition.",
            }
        ]
    )
    assert any("Step: Set up addition." in line for line in lines)


def test_symbolic_validators_catch_remainder_and_factorization_errors() -> None:
    assert check_division_remainder("194 / 11 = 17 with a remainder 9") is False
    assert check_division_remainder("194 / 11 = 17 with a remainder 7") is True
    assert sympy_arithmetic_error("194 / 11 = 17 with a remainder 7") is False
    assert check_polynomial_factorization("x^2 - 2*x - 7 = 0; (x - 7)(x + 5) = 0") is False


def test_binary_first_error_argmax_and_group_sampling() -> None:
    rows = pd.DataFrame(
        [
            {"problem_id": "a", "step_index": 1, "label": "correct"},
            {"problem_id": "a", "step_index": 2, "label": "incorrect"},
            {"problem_id": "b", "step_index": 1, "label": "correct"},
            {"problem_id": "b", "step_index": 2, "label": "correct"},
        ]
    )
    metrics = first_error_metrics(rows, [0.2, 0.8, 0.1, 0.3], threshold=0.5, strategy="argmax")
    assert metrics["first_error_accuracy"] == 1.0
    sampled = select_problem_groups_by_step_budget(rows, 3, seed=7)
    assert all(len(group) == 2 for _, group in sampled.groupby("problem_id"))


def test_binary_text_uses_only_immediate_previous_step() -> None:
    text = make_binary_model_text("Solve x.", "old context ||| x = 2", "Therefore x = 2")
    assert "x = 2" in text
    assert "old context" not in text


def test_target_adaptation_calibration_and_test_groups_are_disjoint() -> None:
    rows = pd.DataFrame(
        [
            {"problem_id": f"p-{problem}", "step_index": step, "label": "correct"}
            for problem in range(20)
            for step in (1, 2)
        ]
    )
    adaptation, calibration, test = split_target_domain_groups(rows, seed=42)
    adaptation_ids = set(adaptation["problem_id"])
    calibration_ids = set(calibration["problem_id"])
    test_ids = set(test["problem_id"])
    assert adaptation_ids.isdisjoint(calibration_ids)
    assert adaptation_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)
    assert adaptation_ids | calibration_ids | test_ids == set(rows["problem_id"])


def test_external_eval_saves_first_error_metrics_and_probabilities(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    train_metrics_path = tmp_path / "train_metrics.json"
    eval_path = tmp_path / "processbench.csv"
    eval_metrics_path = tmp_path / "eval_metrics.json"
    confusion_path = tmp_path / "confusion.png"
    predictions_path = tmp_path / "predictions.json"

    save_dataset(build_synthetic_dataset(n=500, seed=19), train_path)
    train_baseline(train_path, model_path, train_metrics_path)

    eval_rows = []
    eval_rows.extend(
        normalize_processbench_record(
            {
                "id": "case-1",
                "problem": "Compute 2+2.",
                "steps": ["Set up addition.", "2+2=5.", "Answer is 5."],
                "label": 1,
            }
        )
    )
    eval_rows.extend(
        normalize_processbench_record(
            {
                "id": "case-2",
                "problem": "Compute 3+3.",
                "steps": ["Set up addition.", "3+3=6.", "Answer is 6."],
                "label": -1,
            }
        )
    )
    write_rows_csv(eval_rows, eval_path)

    metrics = evaluate(
        model_path,
        eval_path,
        eval_metrics_path,
        confusion_path,
        split_eval=False,
        first_error_strategy="threshold",
        predictions_path=predictions_path,
    )
    assert metrics["split"] == "external eval dataset"
    assert metrics["eval_sources"]["processbench"] == len(eval_rows)
    assert "first_error_accuracy" in metrics["tfidf_logreg"]
    assert metrics["tfidf_logreg"]["first_error_strategy"] == "threshold"
    assert "processbench" in metrics["metrics_by_source"]
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert predictions
    assert {"p_correct", "p_incorrect", "p_suspicious"}.issubset(predictions[0])


def test_prm800k_to_processbench_experiment_writes_outputs(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    model_path = tmp_path / "model.joblib"
    train_metrics_path = tmp_path / "train_metrics.json"
    eval_metrics_path = tmp_path / "eval_metrics.json"
    confusion_path = tmp_path / "confusion.png"
    predictions_path = tmp_path / "predictions.json"
    error_analysis_path = tmp_path / "error_analysis.md"
    combined_report_path = tmp_path / "combined.json"

    save_dataset(build_synthetic_dataset(n=500, seed=23), train_path)
    write_rows_csv(
        normalize_processbench_record(
            {
                "id": "case-1",
                "problem": "Compute 2+2.",
                "steps": ["Set up addition.", "2+2=5.", "Answer is 5."],
                "label": 1,
            }
        ),
        eval_path,
    )

    result = run_experiment(
        train_dataset=train_path,
        eval_dataset=eval_path,
        model_path=model_path,
        train_metrics_path=train_metrics_path,
        eval_metrics_path=eval_metrics_path,
        combined_report_path=combined_report_path,
        confusion_matrix_path=confusion_path,
        predictions_path=predictions_path,
        error_analysis_path=error_analysis_path,
        first_error_strategy="hard_label",
    )
    assert result["experiment"] == "prm800k_to_processbench"
    assert model_path.exists()
    assert eval_metrics_path.exists()
    assert combined_report_path.exists()
    assert predictions_path.exists()
    assert error_analysis_path.exists()
    assert "False Positive Examples" in error_analysis_path.read_text(encoding="utf-8")
