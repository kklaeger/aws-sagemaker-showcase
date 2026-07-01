"""
Training utilities for the AWS SageMaker Showcase project.

This module contains reusable model training logic for the credit risk
classification workflow.
"""
from typing import Any

from xgboost import XGBClassifier

import mlflow

import pandas as pd
import argparse


from itertools import product
from src.evaluate import evaluate_model

TARGET_COLUMN = "class"

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


def load_datasets(
    train_path: str,
    validation_path: str,
    test_path: str | None = None,
) -> tuple:
    """
    Loads the training, validation, and optionally test datasets.

    Parameters:
        train_path (str):
            Path to the training dataset.
        validation_path (str):
            Path to the validation dataset.
        test_path (str | None):
            Optional path to the test dataset.

    Returns:
        tuple:
            (train_df, validation_df, test_df)

            If no test dataset is provided, test_df is None.
    """
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    test_df = None
    if test_path is not None:
        test_df = pd.read_csv(test_path)

    return train_df, validation_df, test_df



def split_features_target(df, target_column):
    """
    Splits a dataframe into features and target.

    Parameters:
        df (pd.DataFrame):
            Input dataframe.
        target_column (str):
            Name of the target column.

    Returns:
        X (pd.DataFrame):
            Feature matrix.
        y (pd.Series):
            Target vector.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

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


def connect_to_mlflow_tracking_server(
    tracking_server_arn: str,
    experiment_name: str,
) -> None:
    """
    Connects to the SageMaker MLflow Tracking Server and selects the experiment.

    Parameters:
        tracking_server_arn (str):
            ARN of the SageMaker MLflow Tracking Server.
        experiment_name (str):
            Name of the MLflow experiment.
    """
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)


def track_baseline_experiment(
    model,
    validation_accuracy: float,
    validation_roc_auc: float,
    run_name: str = "baseline-xgboost",
) -> None:
    """
    Tracks the baseline XGBoost experiment using MLflow.

    Parameters:
        model:
            Trained XGBoost model.
        validation_accuracy (float):
            Validation accuracy.
        validation_roc_auc (float):
            Validation ROC AUC score.
        run_name (str):
            Name of the MLflow run.

    Returns:
        None
    """
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            }
        )

        mlflow.log_metric(
            "validation_accuracy",
            validation_accuracy,
        )

        mlflow.log_metric(
            "validation_roc_auc",
            validation_roc_auc,
        )

        mlflow.xgboost.log_model(
            model,
            name="xgboost_model",
        )


def track_hyperparameter_tuning(
    X_train,
    y_train,
    X_validation,
    y_validation,
    param_grid: dict,
):
    """
    Performs hyperparameter tuning with MLflow experiment tracking.

    Parameters:
        X_train:
            Training feature matrix.
        y_train:
            Training target vector.
        X_validation:
            Validation feature matrix.
        y_validation:
            Validation target vector.
        param_grid (dict):
            Hyperparameter search space.

    Returns:
        best_model:
            Best performing model.
        best_params (dict):
            Hyperparameters of the best model.
        best_results (dict):
            Evaluation results of the best model.
    """
    best_model = None
    best_params = None
    best_results = None
    best_run_name = None

    best_auc = float("-inf")

    param_combinations = product(
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["n_estimators"],
        param_grid["subsample"],
    )

    for i, (
        max_depth,
        learning_rate,
        n_estimators,
        subsample,
    ) in enumerate(param_combinations, start=1):

        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "random_state": 42,
        }

        with mlflow.start_run(run_name=f"run_{i}"):

            model = train_baseline_model(
                X_train,
                y_train,
                params=params,
            )

            results = evaluate_model(
                model,
                X_validation,
                y_validation,
            )

            mlflow.log_params(params)
            mlflow.log_metric(
                "validation_accuracy",
                results["accuracy"],
            )
            mlflow.log_metric(
                "validation_roc_auc",
                results["roc_auc"],
            )

            mlflow.xgboost.log_model(
                model,
                name="xgboost_model",
            )

        if results["roc_auc"] > best_auc:
            best_auc = results["roc_auc"]
            best_model = model
            best_params = params
            best_results = results
            best_run_name = f"run_{i}"

    return (
        best_model,
        best_params,
        best_results,
        best_auc,
        best_run_name,
    )

def save_model(
    model,
    output_path: str,
) -> None:
    """
    Saves a trained model to disk.

    Parameters:
        model:
            Trained XGBoost model.
        output_path (str):
            Output path for the serialized model.

    Returns:
        None
    """
    import joblib

    joblib.dump(model, output_path)


def run_training(
    train_path: str,
    validation_path: str,
    tracking_server_arn: str,
    experiment_name: str,
    model_output: str,
):
    """
    Executes the complete model training workflow.

    Parameters:
        train_path (str):
            Path to the training dataset.
        validation_path (str):
            Path to the validation dataset.
        tracking_server_arn (str):
            ARN of the SageMaker MLflow Tracking Server.
        experiment_name (str):
            Name of the MLflow experiment.
        model_output (str):
            Output path for the trained model.

    Returns:
        model:
            Trained XGBoost model.
        results (dict):
            Validation metrics.
    """
    train_df, validation_df = load_datasets(
        train_path=train_path,
        validation_path=validation_path,
    )

    X_train, y_train = split_features_target(
        train_df,
        TARGET_COLUMN,
    )

    X_validation, y_validation = split_features_target(
        validation_df,
        TARGET_COLUMN,
    )

    connect_to_mlflow_tracking_server(
        tracking_server_arn,
        experiment_name,
    )

    model = train_baseline_model(
        X_train,
        y_train,
    )

    results = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    track_baseline_experiment(
        model=model,
        validation_accuracy=results["accuracy"],
        validation_roc_auc=results["roc_auc"],
    )

    save_model(
        model,
        model_output,
    )

    return model, results


def parse_args():
    """
    Parses command-line arguments for the training script.

    Returns:
        argparse.Namespace:
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to the training dataset.",
    )

    parser.add_argument(
        "--validation-data",
        type=str,
        required=True,
        help="Path to the validation dataset.",
    )

    parser.add_argument(
        "--tracking-server-arn",
        type=str,
        required=True,
        help="ARN of the SageMaker MLflow Tracking Server.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="credit-risk-classification",
        help="MLflow experiment name.",
    )

    parser.add_argument(
        "--model-output",
        type=str,
        required=True,
        help="Output path for the trained model.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    run_training(
        train_path=args.train_data,
        validation_path=args.validation_data,
        tracking_server_arn=args.tracking_server_arn,
        experiment_name=args.experiment_name,
        model_output=args.model_output,
    )


if __name__ == "__main__":
    main()
