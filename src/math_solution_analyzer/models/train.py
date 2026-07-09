from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from math_solution_analyzer.features import extract_step_features, make_model_text


def train_baseline(dataset_path: Path, model_path: Path, metrics_path: Path, test_size: float = 0.25, seed: int = 42) -> dict:
    df = pd.read_csv(dataset_path)
    X = _build_features_frame(df)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    numeric_features = [col for col in X.columns if col != "model_text"]

    pipeline = Pipeline(
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
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "model": "TF-IDF + numeric features + LogisticRegression",
        "rows": int(len(df)),
        "test_size": test_size,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/step_classification.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/tfidf_logreg.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()
    metrics = train_baseline(args.dataset, args.model, args.metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
