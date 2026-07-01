"""
Evaluation utilities for the AWS SageMaker Showcase project.
"""
import json

import argparse

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


def save_evaluation_results(
    results: dict,
    output_path: str,
) -> None:
    """
    Saves evaluation results as a JSON file.

    Parameters:
        results (dict):
            Dictionary containing the evaluation metrics.
        output_path (str):
            Path where the evaluation results should be stored.

    Returns:
        None
    """
    serializable_results = results.copy()

    with open(output_path, "w") as f:
        json.dump(
            serializable_results,
            f,
            indent=4,
        )


def run_evaluation(
    model_path: str,
    test_data_path: str,
    evaluation_output: str,
):
    """
    Executes the complete model evaluation workflow.

    Parameters:
        model_path (str):
            Path to the trained model.
        test_data_path (str):
            Path to the test dataset.
        evaluation_output (str):
            Output path for the evaluation report.

    Returns:
        dict:
            Evaluation results.
    """
    model = load_model(model_path)

    test_df = pd.read_csv(test_data_path)

    X_test, y_test = split_features_target(
        test_df,
        TARGET_COLUMN,
    )

    results = evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_evaluation_results(
        results,
        evaluation_output,
    )

    return results


def parse_args():
    """
    Parses command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace:
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model.",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to the test dataset.",
    )

    parser.add_argument(
        "--evaluation-output",
        type=str,
        required=True,
        help="Path where the evaluation report should be stored.",
    )

    return parser.parse_args()


def main():
    """
    Executes the evaluation script.
    """
    args = parse_args()

    run_evaluation(
        model_path=args.model,
        test_data_path=args.test_data,
        evaluation_output=args.evaluation_output,
    )


if __name__ == "__main__":
    main()
