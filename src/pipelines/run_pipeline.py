import pandas as pd
import logging
import os
import sys
import mlflow
import yaml

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipelines.data_ingestion import load_data
from src.pipelines.preprocessing import preprocess_data
from src.pipelines.training import train_model
from src.pipelines.evaluation import evaluate_model
from src.pipelines.prediction import predict_churn
from src.pipelines.mlflow_setup import setup_mlflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

def main():
    """Run the entire churn prediction pipeline with MLflow tracking."""
    logging.info("Starting pipeline execution...")
    
    # Set up MLflow
    setup_mlflow()
    
    with mlflow.start_run(run_name="Full_Pipeline"):
        # Create required directories
        directories = [
            "C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/data/processed",
            "C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/models/saved_preprocessing",
            "C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/models/trained_models",
            "C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/visuals",
            "C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/logs"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")

        # Step 1: Load data
        logging.info("Loading data...")
        df = load_data()

        # Step 2: Preprocess data
        logging.info("Preprocessing data...")
        X_train_final, X_test_final, y_train, y_test, X_train = preprocess_data(df)

        # Step 3: Train model
        logging.info("Training model...")
        model = train_model(X_train_final, y_train, X_test_final, y_test)

        # Step 4: Evaluate model
        logging.info("Evaluating model...")
        evaluate_model(model, X_test_final, y_test)

        # Step 5: Predict on new data
        logging.info("Making predictions...")
        new_data = pd.DataFrame({
            'Age': [45, 35, 40, 50, 37],
            'Gender': ['Female', 'Female', 'Male', 'Male', 'Female'],
            'Tenure': [24, 30, 22, 28, 20],
            'UsageFrequency': [15, 20, 18, 22, 19],
            'SupportCalls': [0, 1, 0, 0, 1],
            'PaymentDelay': [0, 0, 0, 0, 0],
            'SubscriptionType': ['Premium', 'Premium', 'Premium', 'Premium', 'Premium'],
            'ContractLength': ['Annual', 'Annual', 'Annual', 'Annual', 'Annual'],
            'TotalSpend': [600, 550, 580, 620, 590],
            'LastInteraction': [30, 25, 28, 32, 29]
        })
        predictions, probabilities = predict_churn(new_data, X_train)
        
        # Log predictions as artifacts
        pd.DataFrame({
            'Predictions': predictions,
            'Probabilities': probabilities
        }).to_csv('predictions.csv', index=False)
        mlflow.log_artifact('predictions.csv')
        
        logging.info(f"Predictions: {predictions}")
        logging.info(f"Probabilities: {probabilities}")
        print("Predictions:", predictions)
        print("Probabilities:", probabilities)

        logging.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()