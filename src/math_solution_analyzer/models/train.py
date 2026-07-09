from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from math_solution_analyzer.features import extract_step_features, make_model_text


def train_baseline(dataset_path: Path, model_path: Path, metrics_path: Path, test_size: float = 0.25, seed: int = 42) -> dict:
    df = pd.read_csv(dataset_path)
    X = _build_features_frame(df)
    y_label = df["label"].astype(str)
    y_error_type = df["error_type"].astype(str)
    groups = df["problem_id"].astype(str)
    sources = df["source"].astype(str) if "source" in df.columns else pd.Series(["unknown"] * len(df))

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y_label, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_label_train, y_label_test = y_label.iloc[train_idx], y_label.iloc[test_idx]
    y_error_train, y_error_test = y_error_type.iloc[train_idx], y_error_type.iloc[test_idx]

    numeric_features = [col for col in X.columns if col != "model_text"]

    base_pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "model_text"),
                        ("num", StandardScaler(with_mean=False), numeric_features),
                    ],
                    remainder="drop",
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    label_model = clone(base_pipeline)
    error_type_model = clone(base_pipeline)
    holdout_label_model = clone(base_pipeline)
    holdout_error_type_model = clone(base_pipeline)
    holdout_label_model.fit(X_train, y_label_train)
    holdout_error_type_model.fit(X_train, y_error_train)

    label_predictions = holdout_label_model.predict(X_test)
    error_predictions = holdout_error_type_model.predict(X_test)

    label_model.fit(X, y_label)
    error_type_model.fit(X, y_error_type)

    metrics = {
        "model": "TF-IDF + numeric features + LogisticRegression",
        "split": "GroupShuffleSplit by problem_id",
        "rows": int(len(df)),
        "train_sources": _source_counts(sources),
        "holdout_train_sources": _source_counts(sources.iloc[train_idx]),
        "holdout_eval_sources": _source_counts(sources.iloc[test_idx]),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_groups": int(groups.iloc[train_idx].nunique()),
        "test_groups": int(groups.iloc[test_idx].nunique()),
        "test_size": test_size,
        "label_accuracy": round(float(accuracy_score(y_label_test, label_predictions)), 4),
        "label_macro_f1": round(float(f1_score(y_label_test, label_predictions, average="macro")), 4),
        "error_type_accuracy": round(float(accuracy_score(y_error_test, error_predictions)), 4),
        "error_type_macro_f1": round(float(f1_score(y_error_test, error_predictions, average="macro")), 4),
        "label_classification_report": classification_report(y_label_test, label_predictions, output_dict=True, zero_division=0),
        "error_type_classification_report": classification_report(y_error_test, error_predictions, output_dict=True, zero_division=0),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "label_model": label_model,
            "error_type_model": error_type_model,
            "split": metrics["split"],
            "numeric_features": numeric_features,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _build_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in df.iterrows():
        previous_steps = str(row.get("previous_steps", "")).split(" ||| ") if pd.notna(row.get("previous_steps")) else []
        current_step = str(row["current_step"])
        problem = str(row["problem"])
        feature_row = extract_step_features(
            problem=problem,
            previous_steps=previous_steps,
            current_step=current_step,
            step_index=int(row["step_index"]),
        )
        feature_row["model_text"] = make_model_text(problem, previous_steps, current_step)
        rows.append(feature_row)
    return pd.DataFrame(rows)


def _source_counts(sources: pd.Series) -> dict[str, int]:
    return {str(source): int(count) for source, count in sources.value_counts().sort_index().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset CSV path. Alias kept for toy/local training.")
    parser.add_argument("--train-dataset", type=Path, default=None, help="Training dataset CSV path.")
    parser.add_argument("--model", type=Path, default=Path("models/tfidf_logreg.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()
    dataset_path = args.train_dataset or args.dataset or Path("data/processed/step_classification.csv")
    metrics = train_baseline(dataset_path, args.model, args.metrics)
    compact = {k: v for k, v in metrics.items() if not k.endswith("_classification_report")}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
