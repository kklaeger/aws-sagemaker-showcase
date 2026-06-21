"""
Training utilities for the AWS SageMaker Showcase project.

This module contains reusable model training logic for the credit risk
classification workflow.
"""
from typing import Any

from xgboost import XGBClassifier


DEFAULT_BASELINE_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}

DEFAULT_PARAM_GRID = {
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [50, 100],
    "subsample": [0.8, 1.0],
}


def train_baseline_model(
    X_train,
    y_train,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """
    Trains a baseline XGBoost classifier on the training dataset.

    The function uses a small default configuration that can be overridden
    with custom parameters.

    Parameters:
        X_train:
            Feature matrix used for training.
        y_train:
            Target vector used for training.
        params (dict[str, Any] | None):
            Optional custom XGBoost parameters.

    Returns:
        model (XGBClassifier):
            Trained XGBoost classifier.
    """
    model_params = DEFAULT_BASELINE_PARAMS.copy()

    if params:
        model_params.update(params)

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    return model
