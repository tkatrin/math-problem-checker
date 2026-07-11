from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from math_solution_analyzer.models.binary_benchmark import (
    BinaryModelConfig,
    apply_symbolic_overrides,
    binary_training_rows,
    build_feature_frame,
    embedding_with_numeric_features,
    encode_texts,
    evaluate_binary_predictions,
    fit_binary_model,
    predict_incorrect_probability,
    select_problem_groups_by_step_budget,
    split_calibration_rows,
    tune_threshold,
)


def run_experiment(
    *,
    train_dataset: Path,
    eval_dataset: Path,
    output_path: Path,
    markdown_path: Path,
    cache_dir: Path,
    sizes: tuple[int, ...] = (10_000, 50_000, 100_000),
    seed: int = 42,
    include_embeddings: bool = True,
    best_model_path: Path | None = None,
) -> dict:
    train_all = binary_training_rows(pd.read_csv(train_dataset))
    eval_all = pd.read_csv(eval_dataset)
    max_train = select_problem_groups_by_step_budget(train_all, max(sizes), seed=seed)
    train_features = build_feature_frame(max_train, cache_path=cache_dir / "prm800k_binary_100k_features_v4.pkl")
    eval_features = build_feature_frame(eval_all, cache_path=cache_dir / "processbench_math_features_v4.pkl")

    scale_results: dict[str, dict] = {}
    scale_context: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    default_config = BinaryModelConfig(family="tfidf_logreg", balance_strategy="class_weight", incorrect_weight=2.0, seed=seed)
    for size in sizes:
        subset = select_problem_groups_by_step_budget(max_train, size, seed=seed)
        subset_features = train_features.loc[subset.index]
        result = _fit_calibrate_evaluate(
            subset,
            subset_features,
            eval_all,
            eval_features,
            config=default_config,
        )
        scale_results[str(size)] = result
        scale_context[size] = (subset, subset_features)

    balance_results = {"class_weight_2": scale_results[str(50_000)]}
    balance_subset, balance_features = scale_context[50_000]
    for incorrect_weight in (1.0, 4.0):
        balance_results[f"class_weight_{incorrect_weight:g}"] = _fit_calibrate_evaluate(
            balance_subset,
            balance_features,
            eval_all,
            eval_features,
            config=BinaryModelConfig(
                family="tfidf_logreg",
                balance_strategy="class_weight",
                incorrect_weight=incorrect_weight,
                seed=seed,
            ),
        )
    for strategy in ("undersample_correct", "oversample_incorrect"):
        balance_results[strategy] = _fit_calibrate_evaluate(
            balance_subset,
            balance_features,
            eval_all,
            eval_features,
            config=BinaryModelConfig(family="tfidf_logreg", balance_strategy=strategy, incorrect_weight=2.0, seed=seed),
        )

    family_results = {"tfidf_logreg": scale_results[str(50_000)]}
    for family in ("tfidf_lightgbm",):
        family_results[family] = _fit_calibrate_evaluate(
            balance_subset,
            balance_features,
            eval_all,
            eval_features,
            config=BinaryModelConfig(family=family, balance_strategy="class_weight", incorrect_weight=2.0, seed=seed),
        )

    target_adapted = _fit_target_adapted(
        scale_context[max(sizes)][0],
        scale_context[max(sizes)][1],
        eval_all,
        eval_features,
        config=default_config,
        model_path=best_model_path,
    )

    if not include_embeddings:
        result = _build_result(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_all=train_all,
            eval_all=eval_all,
            scale_results=scale_results,
            balance_results=balance_results,
            family_results=family_results,
            target_adapted=target_adapted,
            sizes=sizes,
        )
        _save_result(result, output_path, markdown_path)
        return result

    train_embeddings = encode_texts(
        balance_features["model_text"],
        cache_path=cache_dir / "prm800k_binary_50k_e5_small.npy",
    )
    eval_embeddings = encode_texts(
        eval_features["model_text"],
        cache_path=cache_dir / "processbench_math_e5_small.npy",
    )
    family_results["embedding_logreg"] = _fit_calibrate_evaluate(
        balance_subset,
        balance_features,
        eval_all,
        eval_features,
        config=BinaryModelConfig(family="embedding_logreg", balance_strategy="class_weight", incorrect_weight=2.0, seed=seed),
        train_matrix=train_embeddings,
        eval_matrix=eval_embeddings,
    )
    family_results["embedding_lightgbm"] = _fit_calibrate_evaluate(
        balance_subset,
        balance_features,
        eval_all,
        eval_features,
        config=BinaryModelConfig(family="embedding_lightgbm", balance_strategy="class_weight", incorrect_weight=2.0, seed=seed),
        train_matrix=embedding_with_numeric_features(train_embeddings, balance_features),
        eval_matrix=embedding_with_numeric_features(eval_embeddings, eval_features),
    )

    result = _build_result(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_all=train_all,
        eval_all=eval_all,
        scale_results=scale_results,
        balance_results=balance_results,
        family_results=family_results,
        target_adapted=target_adapted,
        sizes=sizes,
    )
    _save_result(result, output_path, markdown_path)
    return result


def run_improved_model(
    *,
    train_dataset: Path,
    eval_dataset: Path,
    output_path: Path,
    cache_dir: Path,
    model_path: Path,
    train_size: int = 50_000,
    seed: int = 42,
) -> dict:
    """Train the practical target-adapted model without the full ablation sweep."""

    train_all = binary_training_rows(pd.read_csv(train_dataset))
    train_df = select_problem_groups_by_step_budget(train_all, train_size, seed=seed)
    eval_df = pd.read_csv(eval_dataset)
    train_features = build_feature_frame(
        train_df,
        cache_path=cache_dir / f"prm800k_binary_{train_size}_features_v4.pkl",
    )
    eval_features = build_feature_frame(
        eval_df,
        cache_path=cache_dir / "processbench_math_features_v4.pkl",
    )
    result = _fit_target_adapted(
        train_df,
        train_features,
        eval_df,
        eval_features,
        config=BinaryModelConfig(
            family="tfidf_logreg",
            balance_strategy="class_weight",
            incorrect_weight=2.0,
            seed=seed,
        ),
        target_repeat=4,
        model_path=model_path,
    )
    result["train_steps"] = int(len(train_df))
    result["model_path"] = str(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _build_result(
    *,
    train_dataset: Path,
    eval_dataset: Path,
    train_all: pd.DataFrame,
    eval_all: pd.DataFrame,
    scale_results: dict,
    balance_results: dict,
    family_results: dict,
    target_adapted: dict,
    sizes: tuple[int, ...],
) -> dict:
    best_scale = scale_results[str(max(sizes))]
    return {
        "experiment": "binary_prm800k_scaling_to_processbench",
        "target": "correct_vs_incorrect; PRM800K suspicious/no_progress excluded from train target; ProcessBench after_error_context excluded from step metrics",
        "train_dataset": str(train_dataset),
        "eval_dataset": str(eval_dataset),
        "train_available_binary_rows": int(len(train_all)),
        "eval_rows": int(len(eval_all)),
        "scale_results": scale_results,
        "balance_results": balance_results,
        "family_results": family_results,
        "target_adapted": target_adapted,
        "first_error_strategies_100k": best_scale["external_strategy_metrics"],
    }


def _save_result(result: dict, output_path: Path, markdown_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(result, markdown_path)


def _fit_calibrate_evaluate(
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_features: pd.DataFrame,
    *,
    config: BinaryModelConfig,
    train_matrix: np.ndarray | None = None,
    eval_matrix: np.ndarray | None = None,
) -> dict:
    fit_df, calibration_df = split_calibration_rows(train_df, seed=config.seed)
    fit_features = train_features.loc[fit_df.index]
    calibration_features = train_features.loc[calibration_df.index]
    fit_matrix = _subset_matrix(train_matrix, train_df.index, fit_df.index)
    calibration_matrix = _subset_matrix(train_matrix, train_df.index, calibration_df.index)

    calibration_model = fit_binary_model(
        fit_features,
        fit_df["label"],
        config,
        embedding_matrix=fit_matrix,
    )
    calibration_input = calibration_matrix if calibration_matrix is not None else calibration_features
    calibration_probability = apply_symbolic_overrides(
        calibration_features,
        predict_incorrect_probability(calibration_model, calibration_input),
    )
    threshold_tuning = tune_threshold(calibration_df, calibration_probability, strategy="threshold")
    argmax_tuning = tune_threshold(calibration_df, calibration_probability, strategy="argmax")

    final_model = fit_binary_model(
        train_features,
        train_df["label"],
        config,
        embedding_matrix=train_matrix,
    )
    eval_input = eval_matrix if eval_matrix is not None else eval_features
    eval_probability = apply_symbolic_overrides(
        eval_features,
        predict_incorrect_probability(final_model, eval_input),
    )
    external_metrics = evaluate_binary_predictions(
        eval_df,
        eval_probability,
        threshold=threshold_tuning["threshold"],
        strategy="threshold",
    )
    strategies = {
        "threshold": evaluate_binary_predictions(
            eval_df, eval_probability, threshold=threshold_tuning["threshold"], strategy="threshold"
        ),
        "hard_label": evaluate_binary_predictions(eval_df, eval_probability, threshold=0.5, strategy="hard_label"),
        "hybrid": evaluate_binary_predictions(
            eval_df, eval_probability, threshold=threshold_tuning["threshold"], strategy="hybrid"
        ),
        "argmax": evaluate_binary_predictions(
            eval_df, eval_probability, threshold=argmax_tuning["threshold"], strategy="argmax"
        ),
    }
    return {
        "family": config.family,
        "balance_strategy": config.balance_strategy,
        "incorrect_weight": config.incorrect_weight,
        "source_steps": int(len(train_df)),
        "source_problems": int(train_df["problem_id"].nunique()),
        "fit_steps_for_calibration": int(len(fit_df)),
        "calibration_steps": int(len(calibration_df)),
        "calibration_thresholds": {"threshold": threshold_tuning, "argmax": argmax_tuning},
        "external_metrics": external_metrics,
        "external_strategy_metrics": strategies,
    }


def _fit_target_adapted(
    prm_df: pd.DataFrame,
    prm_features: pd.DataFrame,
    processbench_df: pd.DataFrame,
    processbench_features: pd.DataFrame,
    *,
    config: BinaryModelConfig,
    target_repeat: int = 4,
    model_path: Path | None = None,
) -> dict:
    from sklearn.model_selection import GroupShuffleSplit

    outer = GroupShuffleSplit(n_splits=1, train_size=0.2, random_state=config.seed)
    adapt_idx, test_idx = next(outer.split(processbench_df, groups=processbench_df["problem_id"].astype(str)))
    adapt_all = processbench_df.iloc[adapt_idx].copy()
    test_df = processbench_df.iloc[test_idx].copy()
    adapt_binary = binary_training_rows(adapt_all)
    adapt_fit, adapt_calibration = split_calibration_rows(adapt_binary, seed=config.seed)

    repeated_adapt_fit = pd.concat([adapt_fit] * target_repeat, ignore_index=True)
    repeated_adapt_fit_features = pd.concat(
        [processbench_features.loc[adapt_fit.index]] * target_repeat,
        ignore_index=True,
    )
    combined_df = pd.concat([prm_df, repeated_adapt_fit], ignore_index=True)
    combined_features = pd.concat(
        [prm_features, repeated_adapt_fit_features],
        ignore_index=True,
    )
    calibration_features = processbench_features.loc[adapt_calibration.index]
    calibration_model = fit_binary_model(combined_features, combined_df["label"], config)
    calibration_probability = apply_symbolic_overrides(
        calibration_features,
        predict_incorrect_probability(calibration_model, calibration_features),
    )
    threshold_tuning = tune_threshold(adapt_calibration, calibration_probability, strategy="threshold")
    argmax_tuning = tune_threshold(adapt_calibration, calibration_probability, strategy="argmax")

    repeated_adapt = pd.concat([adapt_binary] * target_repeat, ignore_index=True)
    repeated_adapt_features = pd.concat(
        [processbench_features.loc[adapt_binary.index]] * target_repeat,
        ignore_index=True,
    )
    final_df = pd.concat([prm_df, repeated_adapt], ignore_index=True)
    final_features = pd.concat(
        [prm_features, repeated_adapt_features],
        ignore_index=True,
    )
    final_model = fit_binary_model(final_features, final_df["label"], config)
    test_features = processbench_features.loc[test_df.index]
    test_probability = apply_symbolic_overrides(
        test_features,
        predict_incorrect_probability(final_model, test_features),
    )
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_type": "binary_step_classifier",
                "label_model": final_model,
                "step_threshold": 0.5,
                "first_error_threshold": threshold_tuning["threshold"],
                "training_protocol": "PRM800K + 20% ProcessBench target adaptation",
            },
            model_path,
        )
    return {
        "protocol": "20% ProcessBench groups for adaptation/calibration, 80% held-out groups for test",
        "adaptation_problems": int(adapt_all["problem_id"].nunique()),
        "target_repeat": target_repeat,
        "test_problems": int(test_df["problem_id"].nunique()),
        "threshold": evaluate_binary_predictions(
            test_df,
            test_probability,
            threshold=threshold_tuning["threshold"],
            strategy="threshold",
        ),
        "hard_label": evaluate_binary_predictions(
            test_df,
            test_probability,
            threshold=0.5,
            strategy="hard_label",
        ),
        "argmax": evaluate_binary_predictions(
            test_df,
            test_probability,
            threshold=argmax_tuning["threshold"],
            strategy="argmax",
        ),
        "calibration_thresholds": {"threshold": threshold_tuning, "argmax": argmax_tuning},
    }


def _subset_matrix(matrix: np.ndarray | None, source_index: pd.Index, requested_index: pd.Index) -> np.ndarray | None:
    if matrix is None:
        return None
    positions = pd.Series(np.arange(len(source_index)), index=source_index).loc[requested_index].to_numpy()
    return matrix[positions]


def _write_markdown(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Binary PRM800K Scaling -> ProcessBench",
        "",
        "The supervised target is only `correct` vs `incorrect`. PRM800K `no_progress` and ProcessBench `after_error_context` are excluded from step-level classification metrics.",
        "",
        "## Data Scale",
        "",
        "| Train size | Label macro-F1 | Incorrect recall | First-error acc | Tuned threshold |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for size, item in result["scale_results"].items():
        metrics = item["external_metrics"]
        threshold = item["calibration_thresholds"]["threshold"]["threshold"]
        lines.append(
            f"| {item['source_steps']:,} | {metrics['label_macro_f1']:.4f} | {metrics['incorrect_recall']:.4f} | "
            f"{metrics['first_error_accuracy']:.4f} | {threshold:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Balancing",
            "",
            "| Strategy | Label macro-F1 | Incorrect recall | First-error acc |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for strategy, item in result["balance_results"].items():
        metrics = item["external_metrics"]
        lines.append(
            f"| {strategy} | {metrics['label_macro_f1']:.4f} | {metrics['incorrect_recall']:.4f} | {metrics['first_error_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Model Families",
            "",
            "| Family | Label macro-F1 | Incorrect recall | First-error acc |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for family, item in result["family_results"].items():
        metrics = item["external_metrics"]
        lines.append(
            f"| {family} | {metrics['label_macro_f1']:.4f} | {metrics['incorrect_recall']:.4f} | {metrics['first_error_accuracy']:.4f} |"
        )
    adapted = result["target_adapted"]
    lines.extend(
        [
            "",
            "## Target-Domain Adaptation",
            "",
            adapted["protocol"],
            "",
            "| Strategy | Label macro-F1 | Incorrect recall | First-error acc |",
            "| --- | ---: | ---: | ---: |",
            f"| threshold | {adapted['threshold']['label_macro_f1']:.4f} | {adapted['threshold']['incorrect_recall']:.4f} | {adapted['threshold']['first_error_accuracy']:.4f} |",
            f"| hard_label | {adapted['hard_label']['label_macro_f1']:.4f} | {adapted['hard_label']['incorrect_recall']:.4f} | {adapted['hard_label']['first_error_accuracy']:.4f} |",
            f"| argmax | {adapted['argmax']['label_macro_f1']:.4f} | {adapted['argmax']['incorrect_recall']:.4f} | {adapted['argmax']['first_error_accuracy']:.4f} |",
        ]
    )
    lines.extend(
        [
            "",
            "## First-Error Strategies (largest TF-IDF run)",
            "",
            "| Strategy | First-error acc | First-error macro-F1 | All-correct acc |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for strategy, metrics in result["first_error_strategies_100k"].items():
        lines.append(
            f"| {strategy} | {metrics['first_error_accuracy']:.4f} | {metrics['first_error_macro_f1']:.4f} | {metrics['all_correct_accuracy']} |"
        )
    lines.extend(
        [
            "",
            "Thresholds are tuned on a PRM800K group-held-out calibration split, never on ProcessBench.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, default=Path("data/processed/prm800k_phase2_100mb_steps.csv"))
    parser.add_argument("--eval-dataset", type=Path, default=Path("data/processed/processbench_math_steps.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/binary_prm800k_scaling_to_processbench.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/binary_prm800k_scaling_to_processbench.md"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--best-model", type=Path, default=Path("models/binary_prm800k_target_adapted.joblib"))
    parser.add_argument("--skip-embeddings", action="store_true", help="Run TF-IDF/LightGBM and target adaptation only.")
    parser.add_argument("--improved-only", action="store_true", help="Train only the practical target-adapted model.")
    parser.add_argument("--train-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.improved_only:
        result = run_improved_model(
            train_dataset=args.train_dataset,
            eval_dataset=args.eval_dataset,
            output_path=args.output,
            cache_dir=args.cache_dir,
            model_path=args.best_model,
            train_size=args.train_size,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = run_experiment(
        train_dataset=args.train_dataset,
        eval_dataset=args.eval_dataset,
        output_path=args.output,
        markdown_path=args.markdown,
        cache_dir=args.cache_dir,
        include_embeddings=not args.skip_embeddings,
        best_model_path=args.best_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
