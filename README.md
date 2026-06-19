# AWS SageMaker Showcase

This project is intended as a hands-on learning and showcase project for the complete machine learning lifecycle on AWS SageMaker, from data ingestion to deployment and monitoring.

The project uses the credit-g dataset from OpenML and implements a credit risk classification workflow using XGBoost and AWS SageMaker.

The objective is to predict whether a credit application is likely to be approved based on the available customer and financial information.

The workflow currently covers data ingestion, feature engineering, experiment tracking with MLflow, model training, and model evaluation.

This project is currently a work in progress and is being built incrementally while exploring the AWS SageMaker ecosystem.

## Project Goals

- Demonstrate the AWS SageMaker workflow
- Build an end-to-end machine learning solution on AWS
- Explore SageMaker services and MLOps concepts
- Create a structured project that can be extended over time

## Project Overview

### 01_data_ingestion.ipynb

The first notebook performs the data ingestion process:

- Download the dataset from OpenML
- Inspect the dataset structure
- Perform basic data validation
- Store the raw dataset locally
- Upload the raw dataset to Amazon S3
- Verify the upload

### 02_feature_engineering.ipynb

The second notebook performs feature engineering and Feature Store integration:

- Load the raw dataset from Amazon S3
- Perform data profiling
- Apply preprocessing and feature engineering
- Encode categorical features
- Create a SageMaker Feature Store feature group
- Store engineered features in SageMaker Feature Store
- Save the processed dataset to Amazon S3

### 03_training_and_experiments.ipynb

The third notebook performs model training and experiment tracking:

- Load the processed dataset from Amazon S3
- Inspect the processed dataset
- Split the data into training, validation, and test sets
- Train a baseline XGBoost model
- Evaluate model performance
- Track experiments using MLflow
- Perform hyperparameter tuning
- Compare experiment runs
- Evaluate the best model on the test set
- Save evaluation results to Amazon S3
- Save the trained model artifact to Amazon S3

## AWS Services Used

- Amazon SageMaker Studio
- Amazon SageMaker Feature Store
- Amazon SageMaker MLflow
- Amazon S3
- AWS IAM

## Running the Project

The project is designed to run in AWS SageMaker Studio.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Follow the instructions provided in each notebook.

## License

MIT License
