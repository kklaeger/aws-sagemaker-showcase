"""
Evaluation utilities for the AWS SageMaker Showcase project.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X, y) -> dict:
    """
    Evaluates a trained model on a dataset.

    Returns:
        Dictionary containing evaluation metrics and predictions.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "classification_report": classification_report(y, y_pred),
    }
