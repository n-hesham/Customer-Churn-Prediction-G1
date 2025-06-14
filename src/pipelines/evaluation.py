import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import logging
import yaml
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

with open('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def evaluate_model(model, X_test, y_test):
    """Evaluate model, log results, and track with MLflow."""
    with mlflow.start_run(nested=True):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metric("test_auc_score", auc_score)
        logging.info(f"Test AUC Score: {auc_score:.4f}")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"test_{label}_{metric}", value)
                    logging.info(f"Test {label}_{metric}: {value:.4f}")

        # Generate and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Prediction')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_path = 'C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/visuals/confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()
        
        # Log confusion matrix as artifact
        mlflow.log_artifact(cm_path)
        logging.info(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    model = joblib.load(config['model']['model_path'])
    test_data = pd.read_csv(config['data']['test_processed'])
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn']
    evaluate_model(model, X_test, y_test)