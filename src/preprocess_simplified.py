"""
Preprocessing utilities for the AWS SageMaker Showcase project.

This module contains a small, readable version of the feature-engineering
logic that was originally prototyped in SageMaker Data Wrangler.

The goal is to apply basic preprocessing, one-hot encode categorical features,
and prepare train/validation/test datasets for notebooks and SageMaker Pipelines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "class"

# Columns that are treated as categorical in the German credit dataset.
CATEGORICAL_COLUMNS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
    "other_payment_plans",
]


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes column names compatible with SageMaker Feature Store.

    Allowed special characters are: "-" and "_". All other special
    characters are replaced.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("<=", "le_", regex=False)
        .str.replace(">=", "ge_", regex=False)
        .str.replace("<", "lt_", regex=False)
        .str.replace(">", "gt_", regex=False)
    )

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocessing and feature engineering to the dataset.

    Processing steps:
        - Convert target labels to binary values.
        - One-hot encode categorical features.
        - Preserve numerical features.
        - Make feature names compatible with SageMaker Feature Store.
        - Convert boolean dummy variables to integer values.

    Parameters:
        df (pd.DataFrame):
            Input dataframe containing the raw dataset.

    Returns:
        processed_df (pd.DataFrame):
            Processed dataframe ready for model training.
    """
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    # Map target labels to a binary target.
    # In the credit-g dataset, 'good' is mapped to 1 and 'bad' to 0.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"good": 1, "bad": 0})

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains values other than 'good' and 'bad'."
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")

    # One-hot encode categorical columns.
    # Missing values were not observed during profiling, so no imputation is required here.
    existing_categorical = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    df = pd.get_dummies(df, columns=existing_categorical, drop_first=False)

    # Make feature names compatible with SageMaker Feature Store.
    df = sanitize_feature_names(df)

    # Convert boolean dummy variables to integers (0/1).
    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype("int64")

    return df


def split_data(
    processed_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a processed dataframe into train, validation, and test datasets.

    validation_size and test_size are interpreted as shares of the full dataset.
    For example, validation_size=0.2 and test_size=0.2 create a 60/20/20 split.

    Returns:
        Tuple containing train_df, validation_df, and test_df.
    """
    if target_column not in processed_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be smaller than 1.")

    stratify_values = processed_df[target_column] if stratify else None

    train_validation_df, test_df = train_test_split(
        processed_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    validation_fraction_of_remaining_data = validation_size / (1 - test_size)
    train_validation_stratify_values = (
        train_validation_df[target_column] if stratify else None
    )

    train_df, validation_df = train_test_split(
        train_validation_df,
        test_size=validation_fraction_of_remaining_data,
        random_state=random_state,
        stratify=train_validation_stratify_values,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_output: str | Path,
    validation_output: str | Path,
    test_output: str | Path,
    index: bool = False,
) -> None:
    """
    Writes train, validation, and test datasets to local CSV files.

    This function intentionally writes only local files. In SageMaker Pipelines,
    ProcessingOutput uploads these local files to S3 automatically.
    """
    output_paths = {
        "train": Path(train_output),
        "validation": Path(validation_output),
        "test": Path(test_output),
    }

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_paths["train"], index=index)
    validation_df.to_csv(output_paths["validation"], index=index)
    test_df.to_csv(output_paths["test"], index=index)


def load_input_data(input_data: str | Path) -> pd.DataFrame:
    """
    Loads input data from a CSV file or from a directory containing one CSV file.

    SageMaker Processing usually mounts inputs as local files or directories under
    /opt/ml/processing/input. Supporting both forms makes the script easier to
    use locally and inside SageMaker.
    """
    input_path = Path(input_data)

    if input_path.is_file():
        return pd.read_csv(input_path)

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in input directory: {input_path}")

        if len(csv_files) > 1:
            raise ValueError(
                f"Expected one CSV file in input directory, found {len(csv_files)}: "
                f"{csv_files}"
            )

        return pd.read_csv(csv_files[0])

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_preprocessing(
    input_data: str | Path,
    train_output: str | Path,
    validation_output: str | Path,
    test_output: str | Path,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Runs the full preprocessing workflow and writes local output files.
    """
    df = load_input_data(input_data)
    processed_df = preprocess_dataframe(df)

    train_df, validation_df, test_df = split_data(
        processed_df,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    save_splits(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        train_output=train_output,
        validation_output=validation_output,
        test_output=test_output,
    )

    print("Preprocessing completed successfully.")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments for local execution and SageMaker Processing.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/credit-g.csv",
        help="Local path to the raw input CSV file or input directory.",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="/opt/ml/processing/train/train.csv",
        help="Local path where the train CSV should be written.",
    )
    parser.add_argument(
        "--validation-output",
        type=str,
        default="/opt/ml/processing/validation/validation.csv",
        help="Local path where the validation CSV should be written.",
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="/opt/ml/processing/test/test.csv",
        help="Local path where the test CSV should be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Share of the full dataset used for the test dataset.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Share of the full dataset used for the validation dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state used for reproducible train/validation/test splits.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point used by SageMaker Processing and local command line execution.
    """
    args = parse_args()

    run_preprocessing(
        input_data=args.input_data,
        train_output=args.train_output,
        validation_output=args.validation_output,
        test_output=args.test_output,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
