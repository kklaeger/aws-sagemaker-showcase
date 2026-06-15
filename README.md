# AWS SageMaker Showcase

This project is intended as a hands-on learning and showcase project for the complete machine learning lifecycle on AWS SageMaker, from data ingestion
to deployment and monitoring.

The project uses the credit-g dataset from OpenML and implements a credit-risk classification workflow with XGBoost.

The objective is to predict whether a credit application is likely to be approved based on the available customer and financial information.

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

## Running the Project

The project is designed to run in AWS SageMaker Studio.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Follow the instructions provided in each notebook.

## License

MIT License
