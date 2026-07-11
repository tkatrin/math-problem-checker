from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

import pandas as pd

from math_solution_analyzer.experiments.binary_prm800k_scaling import (
    _fit_calibrate_evaluate,
    _fit_target_adapted,
    split_target_domain_groups,
)
from math_solution_analyzer.models.binary_benchmark import (
    BinaryModelConfig,
    binary_training_rows,
    build_feature_frame,
    select_problem_groups_by_step_budget,
)


DEFAULT_SEEDS = (13, 42, 77, 123, 2026)
ABLATION_VARIANTS = (
    "binary_current_only_with_position",
    "local_context",
    "remove_position_features",
    "soft_symbolic_features",
    "deterministic_symbolic_overrides",
)


def run_single_seed(
    *,
    train_dataset: Path,
    eval_dataset: Path,
    output_path: Path,
    cache_dir: Path,
    train_size: int,
    seed: int,
) -> dict:
    train_all = binary_training_rows(pd.read_csv(train_dataset))
    train_df = select_problem_groups_by_step_budget(train_all, train_size, seed=42)
    eval_df = pd.read_csv(eval_dataset)
    train_features = build_feature_frame(
        train_df,
        context_steps=1,
        cache_path=cache_dir / f"prm800k_binary_{train_size}_context1_v5.pkl",
    )
    eval_features = build_feature_frame(
        eval_df,
        context_steps=1,
        cache_path=cache_dir / "processbench_math_context1_v5.pkl",
    )
    result = _fit_target_adapted(
        train_df,
        train_features,
        eval_df,
        eval_features,
        config=BinaryModelConfig(
            seed=seed,
            incorrect_weight=2.0,
            include_position_features=False,
            include_symbolic_features=True,
        ),
        target_repeat=4,
        use_symbolic_overrides=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_single_ablation(
    *,
    train_dataset: Path,
    eval_dataset: Path,
    output_path: Path,
    cache_dir: Path,
    train_size: int,
    variant: str,
) -> dict:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    context_steps = 0 if variant == "binary_current_only_with_position" else 1
    include_position = variant in {"binary_current_only_with_position", "local_context"}
    include_symbolic = variant in {"soft_symbolic_features", "deterministic_symbolic_overrides"}
    use_overrides = variant == "deterministic_symbolic_overrides"
    train_all = binary_training_rows(pd.read_csv(train_dataset))
    train_df = select_problem_groups_by_step_budget(train_all, train_size, seed=42)
    eval_df = pd.read_csv(eval_dataset)
    train_features = build_feature_frame(
        train_df,
        context_steps=context_steps,
        cache_path=cache_dir / f"prm800k_binary_{train_size}_context{context_steps}_v5.pkl",
    )
    eval_features = build_feature_frame(
        eval_df,
        context_steps=context_steps,
        cache_path=cache_dir / f"processbench_math_context{context_steps}_v5.pkl",
    )
    _, _, heldout_test = split_target_domain_groups(eval_df, seed=42)
    result = _fit_calibrate_evaluate(
        train_df,
        train_features,
        heldout_test,
        eval_features.loc[heldout_test.index],
        config=BinaryModelConfig(
            seed=42,
            incorrect_weight=2.0,
            include_position_features=include_position,
            include_symbolic_features=include_symbolic,
        ),
        use_symbolic_overrides=use_overrides,
    )
    compact = {
        "variant": variant,
        "label_macro_f1": result["external_metrics"]["label_macro_f1"],
        "incorrect_recall": result["external_metrics"]["incorrect_recall"],
        "first_error_accuracy": result["external_metrics"]["first_error_accuracy"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    return compact


def orchestrate_low_memory(
    *,
    train_dataset: Path,
    eval_dataset: Path,
    output_path: Path,
    markdown_path: Path,
    cache_dir: Path,
    train_size: int,
) -> dict:
    partial_dir = output_path.parent / "robustness_parts"
    partial_dir.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable,
        "-m",
        "math_solution_analyzer.experiments.robustness_ablation",
        "--train-dataset",
        str(train_dataset),
        "--eval-dataset",
        str(eval_dataset),
        "--cache-dir",
        str(cache_dir),
        "--train-size",
        str(train_size),
    ]
    for seed in DEFAULT_SEEDS:
        subprocess.run(
            [*common, "--single-seed", str(seed), "--output", str(partial_dir / f"seed_{seed}.json")],
            check=True,
        )
    for variant in ABLATION_VARIANTS:
        subprocess.run(
            [*common, "--single-ablation", variant, "--output", str(partial_dir / f"ablation_{variant}.json")],
            check=True,
        )
    multi_seed = {
        str(seed): json.loads((partial_dir / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in DEFAULT_SEEDS
    }
    ablations = [
        json.loads((partial_dir / f"ablation_{variant}.json").read_text(encoding="utf-8"))
        for variant in ABLATION_VARIANTS
    ]
    adapted = multi_seed["42"]["threshold"]
    ablations.append(
        {
            "variant": "target_adaptation_10_calibration_10",
            "label_macro_f1": adapted["label_macro_f1"],
            "incorrect_recall": adapted["incorrect_recall"],
            "first_error_accuracy": adapted["first_error_accuracy"],
        }
    )
    aggregate = {
        metric: _aggregate([float(result["threshold"][metric]) for result in multi_seed.values()])
        for metric in ("label_macro_f1", "incorrect_recall", "first_error_accuracy", "first_error_macro_f1")
    }
    report = {
        "experiment": "multi_seed_robustness_and_ablation",
        "protocol": "10% adaptation / disjoint 10% calibration / 80% test by problem_id",
        "train_steps": train_size,
        "seeds": list(DEFAULT_SEEDS),
        "aggregate": aggregate,
        "per_seed": multi_seed,
        "ablation": ablations,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, markdown_path)
    return report


def _aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(values), 4),
        "std": round(stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _write_markdown(report: dict, path: Path) -> None:
    aggregate = report["aggregate"]
    lines = [
        "# Robustness and Ablation",
        "",
        f"Protocol: {report['protocol']}.",
        "",
        "## Multi-seed summary",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, values in aggregate.items():
        lines.append(
            f"| {metric} | {values['mean']:.4f} | {values['std']:.4f} | {values['min']:.4f} | {values['max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Cumulative ablation (seed 42)",
            "",
            "| Variant | Label macro-F1 | Incorrect recall | First-error accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in report["ablation"]:
        lines.append(
            f"| {item['variant']} | {item['label_macro_f1']:.4f} | {item['incorrect_recall']:.4f} | {item['first_error_accuracy']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, default=Path("data/processed/prm800k_phase2_100mb_steps.csv"))
    parser.add_argument("--eval-dataset", type=Path, default=Path("data/processed/processbench_math_steps.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/robustness_ablation.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/robustness_ablation.md"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--single-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-ablation", choices=ABLATION_VARIANTS, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single_seed is not None:
        result = run_single_seed(
            train_dataset=args.train_dataset,
            eval_dataset=args.eval_dataset,
            output_path=args.output,
            cache_dir=args.cache_dir,
            train_size=args.train_size,
            seed=args.single_seed,
        )
        print(json.dumps({"seed": args.single_seed, "threshold": result["threshold"]}, ensure_ascii=False, indent=2))
        return
    if args.single_ablation is not None:
        result = run_single_ablation(
            train_dataset=args.train_dataset,
            eval_dataset=args.eval_dataset,
            output_path=args.output,
            cache_dir=args.cache_dir,
            train_size=args.train_size,
            variant=args.single_ablation,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = orchestrate_low_memory(
        train_dataset=args.train_dataset,
        eval_dataset=args.eval_dataset,
        output_path=args.output,
        markdown_path=args.markdown,
        cache_dir=args.cache_dir,
        train_size=args.train_size,
    )
    print(json.dumps({"aggregate": result["aggregate"], "ablation": result["ablation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
