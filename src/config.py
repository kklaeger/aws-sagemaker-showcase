"""
Central configuration for the AWS SageMaker Showcase project.

This file contains shared settings used across notebooks,
Python modules, and SageMaker pipelines.
"""

# AWS
AWS_REGION = "eu-central-1"

# S3
BUCKET_NAME = "aws-sagemaker-showcase"

RAW_DATA_KEY = "raw/german_credit.csv"
PROCESSED_DATA_KEY = "processed/german_credit_processed.csv"
TRAIN_DATA_KEY = "datasets/german_credit_train.csv"
VALIDATION_DATA_KEY = "datasets/german_credit_validation.csv"
TEST_DATA_KEY = "datasets/german_credit_test.csv"

# Feature Store
FEATURE_GROUP_NAME = "credit-risk-features"
FEATURE_STORE_S3_URI = f"s3://{BUCKET_NAME}/feature-store/"

# Training
TARGET_COLUMN = "class"

# MLflow
MLFLOW_TRACKING_SERVER_ARN = (
    "arn:aws:sagemaker:eu-central-1:591874026136:"
    "mlflow-tracking-server/aws-sagemaker-showcase-mlflow"
)

# Model Artifacts
MODEL_ARTIFACT_NAME = "best_model.pkl"
MODEL_ARTIFACT_KEY = f"models/{MODEL_ARTIFACT_NAME}"

SAGEMAKER_MODEL_ARTIFACT_NAME = "model.tar.gz"
SAGEMAKER_MODEL_ARTIFACT_KEY = (
    f"models/sagemaker/xgboost/{SAGEMAKER_MODEL_ARTIFACT_NAME}"
)

SAGEMAKER_MODEL_METADATA_NAME = "model_metadata.json"
SAGEMAKER_MODEL_METADATA_KEY = (
    f"models/sagemaker/xgboost/{SAGEMAKER_MODEL_METADATA_NAME}"
)

EVALUATION_REPORT_NAME = "evaluation.json"
EVALUATION_REPORT_KEY = f"evaluation/{EVALUATION_REPORT_NAME}"

# SageMaker Model Creation
SAGEMAKER_MODEL_NAME_PREFIX = "german-credit-xgboost-model"
ENDPOINT_CONFIG_NAME_PREFIX = "german-credit-xgboost-endpoint-config"
TEMP_ENDPOINT_NAME_PREFIX = "german-credit-xgboost-test-endpoint"

INFERENCE_INSTANCE_TYPE = "ml.m5.large"

XGBOOST_IMAGE_URI = (
    f"492215442770.dkr.ecr.{AWS_REGION}.amazonaws.com/"
    "sagemaker-xgboost:1.7-1"
)
