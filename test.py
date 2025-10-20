import mlflow
import mlflow.data
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Classification")

# Φόρτωσε dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    iris_df[iris.feature_names], iris_df["target"], test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    # Δημιούργησε MLflow dataset object
    dataset = mlflow.data.from_pandas(iris_df, source="sklearn.datasets.load_iris", name="iris_dataset")
    
    # Δήλωσέ το ως input στο run
    mlflow.log_input(dataset, context="training")
    
    # Ορισμός και εκπαίδευση μοντέλου
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Log των παραμέτρων, μετρικών και μοντέλου
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "random_forest_model")
    
    print(f"✅ Model trained successfully! Run ID: {run.info.run_id}")