import mlflow
import mlflow.data
import pandas as pd
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow.models

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Classification")

# Load dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    iris_df[iris.feature_names], iris_df["target"], test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    # Create MLflow dataset object
    dataset = mlflow.data.from_pandas(iris_df, source="sklearn.datasets.load_iris", name="iris_dataset")
    mlflow.log_input(dataset, context="training")
    
    # Define and train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    
    # Log model with input_example and signature
    input_example = X_test[:2]
    signature = mlflow.models.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        clf,
        name="random_forest_model",  
        input_example=input_example,
        signature=signature
    )

    # Register the model in the Model Registry
    model_uri = f"runs:/{run.info.run_id}/random_forest_model"
    mlflow.register_model(model_uri, "RandomForestClassifier")
    
    print(f"✅ Model trained successfully! Run ID: {run.info.run_id}")