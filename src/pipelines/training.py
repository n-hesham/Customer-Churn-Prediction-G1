import pandas as pd
import lightgbm as lgb
import joblib
import logging
import yaml
import mlflow
import mlflow.lightgbm

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

def train_model(X_train, y_train, X_test, y_test):
    """Train LightGBM model and log to MLflow."""
    with mlflow.start_run(nested=True):
        # Define model parameters
        params = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'num_leaves': 31,
            'max_depth': 10,
            'min_data_in_leaf': 20,
            'lambda_l2': 0.1,
            'is_unbalance': True,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Initialize and train model
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # Save model locally
        joblib.dump(model, config['model']['model_path'])
        logging.info(f"Model trained and saved to {config['model']['model_path']}")
        
        # Log model to MLflow
        mlflow.lightgbm.log_model(model, "lightgbm_model")
        
        return model

if __name__ == "__main__":
    train_data = pd.read_csv(config['data']['train_processed'])
    test_data = pd.read_csv(config['data']['test_processed'])
    X_train = train_data.drop('Churn', axis=1)
    y_train = train_data['Churn']
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn']
    model = train_model(X_train, y_train, X_test, y_test)