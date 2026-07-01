import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set MLflow tracking URI ke lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Telco Customer Churn")

# Load data hasil preprocessing
train_df = pd.read_csv("namadataset_preprocessing/train.csv")
test_df = pd.read_csv("namadataset_preprocessing/test.csv")

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

# Aktifkan autolog
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")