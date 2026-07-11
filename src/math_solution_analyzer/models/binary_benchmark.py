"""Binary cross-dataset baselines for first-error detection.

PRM800K's zero rating means ``no_progress`` while ProcessBench marks steps
after an error as ``after_error_context``.  Those labels are intentionally
excluded from the supervised target here: models learn only correct vs.
incorrect, then rank every ProcessBench step by its error probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from math_solution_analyzer.models.train import _build_features_frame


ModelFamily = Literal["tfidf_logreg", "tfidf_lightgbm", "embedding_logreg", "embedding_lightgbm"]
BalanceStrategy = Literal["none", "class_weight", "undersample_correct", "oversample_incorrect"]
FirstErrorStrategy = Literal["threshold", "hard_label", "hybrid", "argmax"]

POSITION_LEAKAGE_FEATURES = {
    "step_index",
    "previous_step_count",
    "previous_context_chars",
}


@dataclass(frozen=True)
class BinaryModelConfig:
    family: ModelFamily = "tfidf_logreg"
    balance_strategy: BalanceStrategy = "class_weight"
    incorrect_weight: float = 4.0
    seed: int = 42


def binary_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only labels with compatible semantics across PRM800K and ProcessBench."""

    result = df[df["label"].isin(["correct", "incorrect"])].copy()
    if result.empty:
        raise ValueError("Binary training requires rows labeled correct or incorrect.")
    return result


def select_problem_groups_by_step_budget(df: pd.DataFrame, budget: int, *, seed: int = 42) -> pd.DataFrame:
    """Sample whole trajectories until the requested step budget is reached."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    groups = list(df.groupby("problem_id", sort=False))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    selected_indices: list[int] = []
    total = 0
    for _, group in groups:
        selected_indices.extend(group.index.tolist())
        total += len(group)
        if total >= budget:
            break
    return df.loc[selected_indices].sort_index().copy()


def build_feature_frame(df: pd.DataFrame, *, cache_path: Path | None = None, batch_size: int = 5_000) -> pd.DataFrame:
    """Extract and optionally cache TF-IDF-ready text plus handcrafted features."""

    if cache_path is not None and cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if list(cached.index) == list(df.index):
            return cached
    batches: list[pd.DataFrame] = []
    for start in range(0, len(df), batch_size):
        # Full symbolic equivalence parsing is valuable in the interactive
        # checker, but far too costly for a 100k-row baseline sweep. The fast
        # frame keeps arithmetic and targeted symbolic-validator flags.
        batch = _build_features_frame(df.iloc[start : start + batch_size], expensive_symbolic=False)
        batch.index = df.index[start : start + batch_size]
        batch["model_text"] = [
            make_binary_model_text(str(problem), str(previous), str(step))
            for problem, previous, step in zip(
                df.iloc[start : start + batch_size]["problem"],
                df.iloc[start : start + batch_size]["previous_steps"],
                df.iloc[start : start + batch_size]["current_step"],
            )
        ]
        batches.append(batch)
    features = pd.concat(batches, axis=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_pickle(cache_path)
    return features


def make_binary_model_text(problem: str, previous_steps: str, current_step: str) -> str:
    """Use compact local evidence and avoid PRM trajectory-position leakage."""

    compact_problem = " ".join(problem.split())[:600]
    previous = "" if previous_steps.lower() == "nan" else previous_steps.split(" ||| ")[-1]
    compact_previous = " ".join(previous.split())[:500]
    compact_step = " ".join(current_step.split())[:900]
    return f"[PROBLEM] {compact_problem} [PREVIOUS] {compact_previous} [STEP] {compact_step}"


def split_calibration_rows(df: pd.DataFrame, *, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, calibration_idx = next(splitter.split(df, groups=df["problem_id"].astype(str)))
    return df.iloc[train_idx].copy(), df.iloc[calibration_idx].copy()


def fit_binary_model(
    features: pd.DataFrame,
    labels: pd.Series,
    config: BinaryModelConfig,
    *,
    embedding_matrix: np.ndarray | None = None,
) -> object:
    """Fit one binary baseline. Label 1 denotes an incorrect step."""

    target = (labels.astype(str) == "incorrect").astype(int).to_numpy()
    train_features, train_target = _apply_balance(features, target, config, embedding_matrix)

    if config.family.startswith("tfidf"):
        numeric_features = [
            column
            for column in features.columns
            if column != "model_text" and column not in POSITION_LEAKAGE_FEATURES
        ]
        classifier = _make_classifier(config)
        model = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "text",
                                TfidfVectorizer(
                                    ngram_range=(1, 2),
                                    min_df=3,
                                    max_df=0.995,
                                    max_features=30_000,
                                    sublinear_tf=True,
                                    strip_accents="unicode",
                                    dtype=np.float32,
                                ),
                                "model_text",
                            ),
                            ("num", StandardScaler(with_mean=False), numeric_features),
                        ],
                        remainder="drop",
                    ),
                ),
                ("clf", classifier),
            ]
        )
        model.fit(train_features, train_target)
        return model

    if embedding_matrix is None:
        raise ValueError("embedding_matrix is required for embedding model families")
    matrix = train_features if isinstance(train_features, np.ndarray) else embedding_matrix
    if config.family == "embedding_logreg":
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", _make_classifier(config)),
            ]
        )
    else:
        model = _make_classifier(config)
    model.fit(matrix, train_target)
    return model


def predict_incorrect_probability(model: object, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classes = list(model.named_steps["clf"].classes_)
    incorrect_index = classes.index(1)
    return np.asarray(probabilities[:, incorrect_index], dtype=float)


def encode_texts(texts: pd.Series, *, cache_path: Path, model_name: str = "intfloat/e5-small-v2") -> np.ndarray:
    """Create normalized sentence embeddings and persist them outside Git."""

    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == len(texts):
            return cached
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install sentence-transformers to run embedding baselines.") from exc
    encoder = SentenceTransformer(model_name)
    prefixed = [f"query: {text}" for text in texts.astype(str)]
    vectors = encoder.encode(prefixed, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return np.asarray(vectors, dtype=np.float32)


def embedding_with_numeric_features(embeddings: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    numeric = features.drop(columns=["model_text", *POSITION_LEAKAGE_FEATURES]).to_numpy(dtype=np.float32)
    return np.hstack([embeddings, numeric])


def apply_symbolic_overrides(features: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    """Promote deterministic validator failures without replacing the ML score."""

    symbolic_columns = [
        "sympy_error",
        "sympy_equivalence_error",
        "derivative_check_error",
        "linear_solution_check_error",
        "probability_fraction_warning",
        "division_remainder_error",
        "polynomial_factorization_error",
    ]
    available = [column for column in symbolic_columns if column in features.columns]
    if not available:
        return probabilities
    result = np.asarray(probabilities, dtype=float).copy()
    deterministic_error = features[available].max(axis=1).to_numpy(dtype=bool)
    result[deterministic_error] = np.maximum(result[deterministic_error], 0.99)
    return result


def evaluate_binary_predictions(
    df: pd.DataFrame,
    incorrect_probability: np.ndarray,
    *,
    threshold: float,
    strategy: FirstErrorStrategy,
) -> dict[str, float | int | str]:
    if len(df) != len(incorrect_probability):
        raise ValueError("df and incorrect_probability must have the same length")
    valid = df["label"].isin(["correct", "incorrect"])
    y_true = (df.loc[valid, "label"] == "incorrect").astype(int).to_numpy()
    y_pred = (incorrect_probability[valid.to_numpy()] >= threshold).astype(int)
    first_error = first_error_metrics(df, incorrect_probability, threshold=threshold, strategy=strategy)
    return {
        "step_rows": int(valid.sum()),
        "label_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "label_macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "incorrect_recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        **first_error,
    }


def first_error_metrics(
    df: pd.DataFrame,
    incorrect_probability: np.ndarray,
    *,
    threshold: float,
    strategy: FirstErrorStrategy,
) -> dict[str, float | int | str]:
    scored = df[["problem_id", "step_index", "label"]].copy()
    scored["p_incorrect"] = incorrect_probability
    true_positions: list[int] = []
    predicted_positions: list[int] = []
    for _, group in scored.sort_values(["problem_id", "step_index"]).groupby("problem_id", sort=False):
        incorrect = group[group["label"] == "incorrect"]
        true_position = int(incorrect.iloc[0]["step_index"]) if not incorrect.empty else -1
        predicted_position = _predict_first_error_position(group, threshold=threshold, strategy=strategy)
        true_positions.append(true_position)
        predicted_positions.append(predicted_position)
    all_correct = [index for index, value in enumerate(true_positions) if value == -1]
    all_correct_accuracy: float | str = "not_applicable"
    if all_correct:
        all_correct_accuracy = round(
            sum(predicted_positions[index] == -1 for index in all_correct) / len(all_correct), 4
        )
    return {
        "first_error_accuracy": round(float(accuracy_score(true_positions, predicted_positions)), 4),
        "first_error_macro_f1": round(float(f1_score(true_positions, predicted_positions, average="macro", zero_division=0)), 4),
        "all_correct_accuracy": all_correct_accuracy,
        "first_error_strategy": strategy,
        "first_error_threshold": round(float(threshold), 2),
    }


def tune_threshold(
    df: pd.DataFrame,
    incorrect_probability: np.ndarray,
    *,
    strategy: FirstErrorStrategy = "argmax",
) -> dict[str, float]:
    candidates = [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
    scored = [
        (
            threshold,
            first_error_metrics(df, incorrect_probability, threshold=threshold, strategy=strategy),
        )
        for threshold in candidates
    ]
    best_threshold, best = max(
        scored,
        key=lambda item: (item[1]["first_error_accuracy"], item[1]["first_error_macro_f1"], -abs(item[0] - 0.5)),
    )
    return {
        "threshold": float(best_threshold),
        "first_error_accuracy": float(best["first_error_accuracy"]),
        "first_error_macro_f1": float(best["first_error_macro_f1"]),
    }


def _predict_first_error_position(group: pd.DataFrame, *, threshold: float, strategy: FirstErrorStrategy) -> int:
    ordered = group.sort_values("step_index")
    if strategy == "argmax":
        best = ordered.loc[ordered["p_incorrect"].idxmax()]
        return int(best["step_index"]) if float(best["p_incorrect"]) >= threshold else -1
    for _, row in ordered.iterrows():
        probability = float(row["p_incorrect"])
        is_error = probability >= threshold
        if strategy == "hard_label":
            is_error = probability >= 0.5
        elif strategy == "hybrid":
            is_error = probability >= threshold or probability >= 0.5
        if is_error:
            return int(row["step_index"])
    return -1


def _apply_balance(
    features: pd.DataFrame,
    target: np.ndarray,
    config: BinaryModelConfig,
    embedding_matrix: np.ndarray | None,
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    correct_indices = np.flatnonzero(target == 0)
    incorrect_indices = np.flatnonzero(target == 1)
    indices = np.arange(len(target))
    if config.balance_strategy == "undersample_correct":
        keep_correct = rng.choice(correct_indices, size=min(len(correct_indices), len(incorrect_indices)), replace=False)
        indices = np.sort(np.concatenate([keep_correct, incorrect_indices]))
    elif config.balance_strategy == "oversample_incorrect":
        extra_incorrect = rng.choice(incorrect_indices, size=max(0, len(correct_indices) - len(incorrect_indices)), replace=True)
        indices = np.concatenate([indices, extra_incorrect])
    selected = embedding_matrix[indices] if embedding_matrix is not None else features.iloc[indices]
    return selected, target[indices]


def _make_classifier(config: BinaryModelConfig):
    class_weight = {0: 1.0, 1: config.incorrect_weight} if config.balance_strategy == "class_weight" else None
    if config.family.endswith("logreg"):
        return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=config.seed)
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install lightgbm to run LightGBM baselines.") from exc
    return LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        class_weight=class_weight,
        random_state=config.seed,
        n_jobs=-1,
        verbosity=-1,
    )
