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
ECALUATION_REPORT_S3_URI = f"s3://{BUCKET_NAME}/{EVALUATION_REPORT_KEY}"

EVALUATION_METADATA_NAME = "evaluation_metadata.json"
EVALUATION_METADATA_KEY = f"evaluation/{EVALUATION_METADATA_NAME}"

EVALUATION_VERSIONED_PREFIX = "evaluation/runs"

# SageMaker Model Creation
SAGEMAKER_MODEL_NAME_PREFIX = "german-credit-xgboost-model"
ENDPOINT_CONFIG_NAME_PREFIX = "german-credit-xgboost-endpoint-config"
TEMP_ENDPOINT_NAME_PREFIX = "german-credit-xgboost-test-endpoint"

INFERENCE_INSTANCE_TYPE = "ml.m5.large"

XGBOOST_IMAGE_URI = (
    f"492215442770.dkr.ecr.{AWS_REGION}.amazonaws.com/"
    "sagemaker-xgboost:1.7-1"
)

# SageMaker Clarify
CLARIFY_FACET_NAME = "age"
CLARIFY_FACET_THRESHOLD = 25

CLARIFY_PROCESSING_INSTANCE_TYPE = "ml.t3.medium"
CLARIFY_PROCESSING_VOLUME_SIZE_GB = 20
CLARIFY_MAX_RUNTIME_SECONDS = 3600

CLARIFY_INPUT_KEY = "clarify/input/clarify_input.csv"
CLARIFY_ANALYSIS_CONFIG_KEY = "clarify/config/analysis_config.json"
CLARIFY_OUTPUT_PREFIX = "clarify/output"

CLARIFY_JOB_NAME_PREFIX = "german-credit-clarify"
CLARIFY_SAMPLE_SIZE = 100

CLARIFY_IMAGE_URI = (
    f"017069133835.dkr.ecr.{AWS_REGION}.amazonaws.com/"
    "sagemaker-clarify-processing:1.0"
)

# SageMaker Model Registry
MODEL_PACKAGE_GROUP_NAME = "german-credit-risk-xgboost"
MODEL_PACKAGE_GROUP_DESCRIPTION = (
    "Model package group for the German Credit Risk XGBoost model."
)

MODEL_APPROVAL_STATUS = "PendingManualApproval"

MODEL_REGISTRY_METADATA_NAME = "model_registry_metadata.json"
MODEL_REGISTRY_METADATA_KEY = f"models/registry/{MODEL_REGISTRY_METADATA_NAME}"

CLARIFY_METADATA_PREFIX = "clarify/metadata/"
