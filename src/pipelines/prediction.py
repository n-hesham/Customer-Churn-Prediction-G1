import pandas as pd
import joblib
import logging
import yaml
from src.pipelines.preprocessing import preprocess_data

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

def predict_churn(new_data_df, X_train):
    """Predict churn for new data."""
    model = joblib.load(config['model']['model_path'])
    X_new_final = preprocess_data(new_data_df, X_train, is_training=False)
    predictions = model.predict(X_new_final)
    probabilities = model.predict_proba(X_new_final)[:, 1]
    logging.info("Predictions generated successfully.")
    return predictions, probabilities

if __name__ == "__main__":
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
    train_data = pd.read_csv(config['data']['train_processed'])
    X_train = train_data.drop('Churn', axis=1)
    predictions, probabilities = predict_churn(new_data, X_train)
    print("Predictions:", predictions)
    print("Probabilities:", probabilities)