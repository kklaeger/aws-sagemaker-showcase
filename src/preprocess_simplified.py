"""Preprocessing utilities for the AWS SageMaker Showcase project.

This module contains a small, readable version of the feature-engineering
logic that was originally prototyped in SageMaker Data Wrangler.

The goal is to apply basic preprocessing and one-hot encode categorical features.
"""
from __future__ import annotations

import boto3
import pandas as pd


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
