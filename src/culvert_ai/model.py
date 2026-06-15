from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from culvert_ai.io import ensure_parent_dir


warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)


DEFAULT_EXCLUDED_FEATURES = {
    "is_culvert",
    "dist_to_known_culvert_m",
    "longitude",
    "latitude",
    "x_m",
    "y_m",
    "culvert_probability",
    "priority_rank",
    "spatial_block_id",
}


def select_feature_columns(
    table: pd.DataFrame,
    target_column: str = "is_culvert",
    extra_excluded: set[str] | None = None,
) -> list[str]:
    excluded = set(DEFAULT_EXCLUDED_FEATURES)
    excluded.add(target_column)
    if extra_excluded:
        excluded |= extra_excluded

    numeric_columns = table.select_dtypes(include=[np.number]).columns
    return [
        column
        for column in numeric_columns
        if column not in excluded and not column.lower().endswith("_id")
    ]


def train_model(
    features: gpd.GeoDataFrame,
    model_output: str | Path,
    metrics_output: str | Path | None = None,
    importance_output: str | Path | None = None,
    target_column: str = "is_culvert",
    test_size: float = 0.25,
    random_state: int = 42,
    model_family: str = "auto",
    spatial_cv: bool = True,
    spatial_block_size_m: float = 2_500.0,
) -> dict:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

    if target_column not in features.columns:
        raise ValueError(f"Target column not found: {target_column}")

    y = features[target_column].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training data needs at least one positive and one negative example.")

    feature_columns = select_feature_columns(features, target_column=target_column)
    if not feature_columns:
        raise ValueError("No numeric feature columns were found for training.")

    x = _prepare_features(features, feature_columns)

    model_candidates = _candidate_models(random_state)
    if model_family != "auto":
        if model_family not in model_candidates:
            valid = ", ".join(["auto", *sorted(model_candidates)])
            raise ValueError(f"Unknown model family '{model_family}'. Valid options: {valid}")
        model_candidates = {model_family: model_candidates[model_family]}

    metrics = {
        "feature_columns": feature_columns,
        "rows": int(len(features)),
        "selection_metric": "cross_validated_average_precision",
    }
    class_counts = y.value_counts().to_dict()
    metrics["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}
    metrics["model_comparison"] = _compare_models(model_candidates, x, y)

    can_split = len(features) >= 8 and y.value_counts().min() >= 2
    selected_name = _select_model(metrics["model_comparison"])
    selected_model = clone(model_candidates[selected_name])
    metrics["selected_model"] = selected_name

    if can_split:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        metrics["random_holdout"] = _fit_and_score(
            selected_model,
            x_train,
            x_test,
            y_train,
            y_test,
            top_k=(5, 10, 25),
        )

        if spatial_cv:
            spatial_holdout = _spatial_holdout_score(
                selected_model,
                features,
                x,
                y,
                test_size=test_size,
                random_state=random_state,
                block_size_m=spatial_block_size_m,
            )
            if spatial_holdout:
                metrics["spatial_holdout"] = spatial_holdout
    else:
        metrics["note"] = "Dataset too small for a reliable holdout split; trained on all rows."

    final_model = clone(model_candidates[selected_name])
    final_model.fit(x, y)
    importances = _feature_importance(final_model, x, y, feature_columns, random_state)
    metrics["feature_importance"] = importances

    bundle = {
        "model": final_model,
        "model_family": selected_name,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "fill_value": -9999.0,
        "random_state": random_state,
        "training_rows": int(len(features)),
        "class_counts": metrics["class_counts"],
    }
    ensure_parent_dir(model_output)
    joblib.dump(bundle, model_output)

    if metrics_output:
        ensure_parent_dir(metrics_output)
        Path(metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if importance_output:
        ensure_parent_dir(importance_output)
        pd.DataFrame(importances).to_csv(importance_output, index=False)

    return metrics


def predict_culvert_probability(
    features: gpd.GeoDataFrame,
    model_path: str | Path,
) -> gpd.GeoDataFrame:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    x = _prepare_features(features, feature_columns, fill_value=bundle.get("fill_value", -9999.0))

    result = features.copy()
    result["culvert_probability"] = model.predict_proba(x)[:, 1]
    result = result.sort_values("culvert_probability", ascending=False).reset_index(drop=True)
    result["priority_rank"] = np.arange(1, len(result) + 1)
    result["priority_percentile"] = 1.0 - ((result["priority_rank"] - 1) / max(len(result), 1))
    result["priority_bucket"] = pd.cut(
        result["culvert_probability"],
        bins=[-0.01, 0.35, 0.65, 0.85, 1.0],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)
    return result


def _candidate_models(random_state: int) -> dict:
    return {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "regularized_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                random_state=random_state,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=random_state,
        ),
    }


def _compare_models(models: dict, x: pd.DataFrame, y: pd.Series) -> dict:
    min_class_count = int(y.value_counts().min())
    if len(x) < 8 or min_class_count < 2:
        return {
            name: {
                "note": "Not enough labeled data for cross-validation.",
                "mean_average_precision": None,
            }
            for name in models
        }

    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    }

    comparison = {}
    for name, estimator in models.items():
        try:
            scores = cross_validate(
                estimator,
                x,
                y,
                cv=cv,
                scoring=scoring,
                error_score=np.nan,
                n_jobs=None,
            )
            comparison[name] = {
                "folds": int(n_splits),
                "mean_roc_auc": _nan_mean(scores["test_roc_auc"]),
                "mean_average_precision": _nan_mean(scores["test_average_precision"]),
                "mean_f1": _nan_mean(scores["test_f1"]),
                "mean_precision": _nan_mean(scores["test_precision"]),
                "mean_recall": _nan_mean(scores["test_recall"]),
            }
        except Exception as exc:
            comparison[name] = {
                "error": str(exc),
                "mean_average_precision": None,
                "mean_f1": None,
            }

    return comparison


def _select_model(comparison: dict) -> str:
    non_baseline = {
        name: result for name, result in comparison.items() if not name.startswith("baseline")
    }
    candidates = non_baseline or comparison

    def score(item) -> tuple[float, float]:
        _name, result = item
        avg_precision = result.get("mean_average_precision")
        f1 = result.get("mean_f1")
        return (
            float(avg_precision) if avg_precision is not None and not np.isnan(avg_precision) else -1.0,
            float(f1) if f1 is not None and not np.isnan(f1) else -1.0,
        )

    return max(candidates.items(), key=score)[0]


def _fit_and_score(model, x_train, x_test, y_train, y_test, top_k: tuple[int, ...]) -> dict:
    fitted = clone(model)
    fitted.fit(x_train, y_train)
    probabilities = fitted.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = _classification_metrics(y_test, predictions, probabilities)
    metrics["rows_train"] = int(len(x_train))
    metrics["rows_test"] = int(len(x_test))
    metrics["top_k"] = _top_k_metrics(y_test, probabilities, top_k)
    return metrics


def _spatial_holdout_score(
    model,
    features: gpd.GeoDataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    block_size_m: float,
) -> dict | None:
    groups = _spatial_blocks(features, block_size_m)
    if groups.nunique() < 3:
        return None

    splitter = GroupShuffleSplit(n_splits=25, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(x, y, groups):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        metrics = _fit_and_score(
            model,
            x.iloc[train_idx],
            x.iloc[test_idx],
            y_train,
            y_test,
            top_k=(5, 10, 25),
        )
        metrics["spatial_block_size_m"] = float(block_size_m)
        metrics["train_blocks"] = int(groups.iloc[train_idx].nunique())
        metrics["test_blocks"] = int(groups.iloc[test_idx].nunique())
        return metrics

    return {
        "note": "Spatial holdout skipped because no split had both positive and negative labels.",
        "spatial_block_size_m": float(block_size_m),
        "blocks": int(groups.nunique()),
    }


def _spatial_blocks(features: gpd.GeoDataFrame, block_size_m: float) -> pd.Series:
    if "x_m" in features.columns and "y_m" in features.columns:
        x_coord = features["x_m"].astype(float)
        y_coord = features["y_m"].astype(float)
    else:
        x_coord = features.geometry.x
        y_coord = features.geometry.y

    x_block = np.floor(x_coord / block_size_m).astype(int)
    y_block = np.floor(y_coord / block_size_m).astype(int)
    return pd.Series(x_block.astype(str) + "_" + y_block.astype(str), index=features.index)


def _feature_importance(
    model,
    x: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    random_state: int,
) -> list[dict]:
    values = None
    method = None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        method = "model_feature_importance"
    elif hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, "coef_"):
            values = np.abs(final_step.coef_[0])
            method = "absolute_logistic_coefficient"

    if values is None and y.nunique() > 1:
        try:
            permutation = permutation_importance(
                model,
                x,
                y,
                scoring="average_precision",
                n_repeats=10,
                random_state=random_state,
                n_jobs=1,
            )
            values = permutation.importances_mean
            method = "permutation_average_precision"
        except Exception:
            values = np.zeros(len(feature_columns))
            method = "unavailable"

    if values is None:
        values = np.zeros(len(feature_columns))
        method = "unavailable"

    total = float(np.sum(np.abs(values)))
    rows = []
    for column, value in sorted(
        zip(feature_columns, values, strict=False), key=lambda pair: abs(pair[1]), reverse=True
    ):
        rows.append(
            {
                "feature": column,
                "importance": float(value),
                "normalized_importance": float(abs(value) / total) if total else 0.0,
                "method": method,
            }
        )
    return rows


def _prepare_features(
    table: pd.DataFrame,
    feature_columns: list[str],
    fill_value: float = -9999.0,
) -> pd.DataFrame:
    prepared = table.copy()
    for column in feature_columns:
        if column not in prepared.columns:
            prepared[column] = np.nan

    return (
        prepared[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_value)
        .astype(float)
    )


def _classification_metrics(y_true, y_pred, y_probability) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability))
    return metrics


def _top_k_metrics(y_true, y_probability, top_k: tuple[int, ...]) -> list[dict]:
    ranked = pd.DataFrame({"target": list(y_true), "probability": list(y_probability)}).sort_values(
        "probability", ascending=False
    )
    positives = int(ranked["target"].sum())
    rows = []
    for k in top_k:
        k_clamped = min(k, len(ranked))
        if k_clamped <= 0:
            continue
        top = ranked.head(k_clamped)
        hits = int(top["target"].sum())
        rows.append(
            {
                "k": int(k_clamped),
                "hits": hits,
                "precision_at_k": float(hits / k_clamped),
                "recall_at_k": float(hits / positives) if positives else 0.0,
            }
        )
    return rows


def _nan_mean(values) -> float | None:
    mean = np.nanmean(values)
    if np.isnan(mean):
        return None
    return float(mean)
