import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Load dataset
iris = load_iris()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Start an MLflow experiment, everything will be logged to mlflow
with mlflow.start_run():

    # Define model and parameters
    n_estimators = 100  # Number of trees
    max_depth = 5
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    
    
    # calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Log model itself
    mlflow.sklearn.log_model(clf, "random_forest_model")
    
    # Register στο Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"
    mlflow.register_model(model_uri, "RandomForestClassifier")

    print(f"Model trained with accuracy: {acc}")