🧠 MLflow → TMF915 AI Model Specification Mapper

📋 Overview

This project provides a dynamic bridge between MLflow tracking runs and TM Forum AI Model Specification (TMF915) format.
It allows any ML model logged in MLflow to be automatically translated into a TMF-compliant JSON specification.

🧩 Core Components

test.py  -> Trains and logs a sample ML model (RandomForest on Iris dataset) in MLflow, including metrics, params, dataset, and model artifacts.

mlflow_to_tmf.py  -> Dynamically fetches an MLflow run via REST API and converts it to TMF915-compliant JSON.

mlflow_to_tmf_api.py  -> FastAPI wrapper providing /map endpoint for mapping MLflow run → TMF JSON.

server_tmf.py  -> Mock TMF server used for validating TMF915 payloads via Swagger UI.

⚙️ Setup Instructions

1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Train a Model and Log to MLflow

python test.py

3️⃣ Run MLflow Tracking Server

mlflow ui --host 127.0.0.1 --port 5000

4️⃣ Run the Mapping API

uvicorn mlflow_to_tmf_api:app --reload --port 8000

5️⃣ Generate TMF JSON

curl -X POST "http://127.0.0.1:8000/map" \
     -H "Content-Type: application/json" \
     -d '{
           "run_id": "YOUR_RUN_ID",
           "tracking_uri": "http://127.0.0.1:5000"
         }'

This returns a TMF-compliant JSON object that can be validated against the TMF915 schema or posted to your TMF AIM API.
