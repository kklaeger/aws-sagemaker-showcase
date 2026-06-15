"""
Central configuration for the AWS SageMaker Showcase project.

This file contains shared settings used across notebooks,
Python modules and SageMaker pipelines.
"""

# AWS Configuration
AWS_REGION = "eu-central-1"

# S3 Configuration
BUCKET_NAME = "aws-sagemaker-showcase"

# Raw dataset uploaded during data ingestion
RAW_DATA_KEY = "raw/german_credit.csv"

# Processed dataset generated during feature engineering
PROCESSED_DATA_KEY = "processed/german_credit_processed.csv"

# Feature Store
FEATURE_GROUP_NAME = "credit-risk-features"
FEATURE_STORE_S3_URI = f"s3://{BUCKET_NAME}/feature-store/"

# Training
TARGET_COLUMN = "class"
