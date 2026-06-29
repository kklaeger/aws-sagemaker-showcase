# AWS SageMaker Showcase

This project is intended as a hands-on learning and showcase project for the complete machine learning lifecycle on AWS SageMaker, from data ingestion to deployment and monitoring.

The project uses the credit-g dataset from OpenML and implements a credit risk classification workflow using XGBoost and AWS SageMaker.

The objective is to predict whether a credit application is likely to be approved based on the available customer and financial information.

The workflow currently covers data ingestion, data processing and feature engineering, experiment tracking with MLflow, model training, model evaluation, SageMaker Pipelines for workflow automation, SageMaker model creation for inference, responsible AI analysis with SageMaker Clarify, model registration with SageMaker Model Registry, and real-time endpoint deployment.

This project is currently a work in progress and is being built incrementally while exploring the AWS SageMaker ecosystem.

## Project Goals

- Demonstrate the AWS SageMaker workflow
- Build an end-to-end machine learning solution on AWS
- Explore SageMaker services and MLOps concepts
- Demonstrate SageMaker Pipelines for automated and reproducible ML workflows
- Create a structured project that can be extended over time

## Prerequisites

Before running the notebooks, the following resources are required:

- An AWS account with access to Amazon SageMaker Studio
- An S3 bucket for project artifacts and datasets
- A SageMaker execution role with permissions for SageMaker, SageMaker Pipelines, SageMaker Processing Jobs, S3, IAM, and MLflow
- A SageMaker MLflow Tracking Server for experiment tracking

## Project Structure

```text
aws-sagemaker-showcase/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training_and_experiments.ipynb
│   ├── 04_model_creation.ipynb
│   ├── 05_clarify.ipynb
│   ├── 06_model_registry.ipynb
│   ├── 07_deployment.ipynb
│   └── 08_pipeline.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocess_simplified.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

The notebooks demonstrate the end-to-end SageMaker workflow, while the src directory contains reusable preprocessing, training, and evaluation utilities.


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
- Split the data into training, validation, and test sets
- Save the processed and the split datasets to Amazon S3

### 03_training_and_experiments.ipynb

The third notebook performs model training and experiment tracking:

- Load the processed dataset from Amazon S3
- Inspect the processed dataset
- Train a baseline XGBoost model
- Evaluate model performance
- Track experiments using MLflow
- Perform hyperparameter tuning
- Compare experiment runs
- Evaluate the best model on the test set
- Save evaluation results to Amazon S3
- Save the trained model artifact to Amazon S3

### 04_model_creation.ipynb

The fourth notebook prepares the locally trained XGBoost model for SageMaker inference:

- Load the trained model artifact from Amazon S3
- Convert the trained XGBoost classifier into a Booster model
- Verify that the converted Booster preserves the prediction behavior
- Create a SageMaker-compatible `model.tar.gz` artifact
- Upload the SageMaker model artifact to Amazon S3
- Create a SageMaker Model using the SageMaker XGBoost inference container
- Deploy a temporary endpoint for a smoke test
- Invoke the temporary endpoint with sample test data
- Compare local and endpoint predictions
- Delete the temporary endpoint and endpoint configuration
- Store model metadata for the next workflow step

### 05_clarify.ipynb

The fifth notebook performs responsible AI analysis with SageMaker Clarify:

- Load the SageMaker Model metadata from Amazon S3
- Prepare a Clarify input dataset from the test dataset
- Create a Clarify analysis configuration
- Run a SageMaker Clarify Processing Job
- Generate bias and explainability results
- Review global SHAP feature importance values
- Review post-training bias metrics for the configured age facet
- Store Clarify outputs and metadata in Amazon S3

### 06_model_registry.ipynb

The sixth notebook registers the model in the SageMaker Model Registry:

- Load SageMaker model metadata from Amazon S3
- Load the evaluation report from Amazon S3
- Load the latest Clarify metadata and analysis result
- Create or reuse a SageMaker Model Package Group
- Register a new Model Package Version
- Attach model quality, bias, and explainability metadata
- Review the registered model package
- Approve the model package for deployment
- Store registry metadata for the deployment workflow

### 07_deployment.ipynb

The seventh notebook deploys the approved model package from SageMaker Model Registry:

- Load registry metadata from Amazon S3
- Verify that the registered model package is approved
- Create a SageMaker Model from the approved model package
- Create an endpoint configuration
- Deploy a SageMaker real-time endpoint
- Invoke the endpoint with sample test records
- Review prediction results
- Delete the endpoint and endpoint configuration to avoid ongoing costs

### 08_sagemaker_pipeline.ipynb

The eighth notebook demonstrates SageMaker Pipelines by converting the data processing workflow into a managed pipeline step:

- Define reusable pipeline parameters
- Configure a SageMaker Processing job
- Run the preprocessing script as a SageMaker ProcessingStep
- Create train, validation, and test datasets inside the pipeline
- Store pipeline-generated artifacts in Amazon S3
- Review the pipeline definition
- Create or update the SageMaker Pipeline
- Start a pipeline execution
- Verify the generated pipeline outputs in Amazon S3

## AWS Services Used

- Amazon SageMaker Studio
- Amazon SageMaker Feature Store
- Amazon SageMaker MLflow
- Amazon SageMaker Model
- Amazon SageMaker Real-Time Inference
- Amazon SageMaker Clarify
- Amazon SageMaker Processing Jobs
- Amazon SageMaker Model Registry
- Amazon SageMaker Pipelines
- Amazon S3
- AWS IAM

## Running the Project

The project is designed to run in AWS SageMaker Studio.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Follow the notebooks in order:

1. `01_data_ingestion.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_training_and_experiments.ipynb`
4. `04_model_creation.ipynb`
5. `05_clarify.ipynb`
6. `06_model_registry.ipynb`
7. `07_deployment.ipynb`

Or use the notebook `08_pipeline.ipynb`.

## Responsible AI Scope

SageMaker Clarify is used to perform an exploratory responsible AI analysis on a small sample of the test dataset.

The Clarify workflow generates global SHAP feature importance values and post-training bias metrics for a configured age facet. The results are intended to demonstrate responsible AI concepts and should not be interpreted as a final fairness assessment or as a production-ready credit decisioning system.

## License

MIT License
