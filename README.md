This repository provides a FastAPI-based service that maps **MLflow run metadata** to the **TMF AI Model Specification (AIM)** format. 
It allows the integration of ML experiment tracking (via MLflow) with TMF-compatible systems and APIs.


├── mlflow_to_tmf.py # Main logic for mapping MLflow to TMF
├── mlflow_to_tmf_api.py # FastAPI routes for mapping
├── server_tmf.py # FastAPI server bootstrap
├── swagger.yaml # OpenAPI spec
├── requirements.txt # Dependencies
├── test.py # Sample script for testing the mapper
