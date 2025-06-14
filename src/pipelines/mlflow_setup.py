import mlflow
import yaml
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

def setup_mlflow():
    """Set up MLflow tracking and experiment."""
    with open('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tracking_uri = config['mlflow']['tracking_uri']
    experiment_name = config['mlflow']['experiment_name']
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        logging.info(f"Created MLflow experiment: {experiment_name}")
    else:
        logging.info(f"Using existing MLflow experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    setup_mlflow()