from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    metrics: dict[str, float | dict]
    test_predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df["season"].dropna().astype(int).unique())
    if len(seasons) < 2:
        split_idx = max(1, int(len(df) * 0.8))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    test_season = seasons[-1]
    train_df = df[df["season"] < test_season].copy()
    test_df = df[df["season"] == test_season].copy()
    return train_df, test_df


def _build_pipeline(numeric_features: Iterable[str], categorical_features: Iterable[str], random_state: int) -> Pipeline:
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    target: str = "home_covers",
) -> ModelArtifacts:
    train_df, test_df = temporal_split(df)

    features = [*numeric_features, *categorical_features]
    X_train = train_df[features]
    y_train = train_df[target].astype(int)
    X_test = test_df[features]
    y_test = test_df[target].astype(int)

    pipeline = _build_pipeline(numeric_features, categorical_features, random_state=random_state)
    pipeline.fit(X_train, y_train)

    test_pred = pipeline.predict(X_test)
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics: dict[str, float | dict] = {
        "train_games": float(len(train_df)),
        "test_games": float(len(test_df)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
    }

    if y_test.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, test_prob))

    metrics["classification_report"] = classification_report(y_test, test_pred, output_dict=True)

    confusion = confusion_matrix(y_test, test_pred)
    metrics["tn"] = float(confusion[0, 0])
    metrics["fp"] = float(confusion[0, 1])
    metrics["fn"] = float(confusion[1, 0])
    metrics["tp"] = float(confusion[1, 1])

    test_predictions = test_df.copy()
    test_predictions[f"pred_{target}"] = test_pred
    test_predictions[f"pred_prob_{target}"] = test_prob

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    return ModelArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        test_predictions=test_predictions,
        feature_importance=feature_importance,
    )


def fit_full_model(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    target: str = "home_covers",
) -> Pipeline:
    features = [*numeric_features, *categorical_features]
    pipeline = _build_pipeline(numeric_features, categorical_features, random_state=random_state)
    pipeline.fit(df[features], df[target].astype(int))
    return pipeline
