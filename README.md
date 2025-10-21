🧠 MLflow → TMF915 AI Model Specification Mapper

📋 Overview
This project provides a dynamic bridge between MLflow tracking runs and TM Forum AI Model Specification (TMF915) format.
It allows any ML model logged in MLflow to be automatically translated into a TMF-compliant JSON specification.

🧩 Core Components
test.py  -> Trains and logs a sample ML model (RandomForest on Iris dataset) in MLflow, including metrics, params, dataset, and model artifacts.

mlflow_to_tmf.py  -> Dynamically fetches an MLflow run via REST API and converts it to TMF915-compliant JSON.

mlflow_to_tmf_api.py  -> FastAPI wrapper providing /map endpoint for mapping MLflow run → TMF JSON.

server_tmf.py  -> Mock TMF server used for validating TMF915 payloads via Swagger UI.
