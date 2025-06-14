import pandas as pd
import logging
import yaml
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import sys
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipelines.preprocessing import preprocess_data
from pipelines.prediction import predict_churn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

# Load configuration
with open('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Pydantic model for input data
class CustomerData(BaseModel):
    Age: int
    Gender: str
    Tenure: int
    UsageFrequency: int
    SupportCalls: int
    PaymentDelay: int
    SubscriptionType: str
    ContractLength: str
    TotalSpend: float
    LastInteraction: int

# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]

# Pydantic model for MLflow run info
class MLflowRun(BaseModel):
    run_id: str
    start_time: str
    status: str
    metrics: dict
    params: dict

@app.on_event("startup")
async def startup_event():
    """Set up MLflow on startup."""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    logging.info("MLflow configured and API started.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: List[CustomerData]):
    """Predict churn for new customer data."""
    try:
        # Convert input data to DataFrame
        new_data = pd.DataFrame([item.dict() for item in data])
        
        # Load training data for preprocessing
        train_data = pd.read_csv(config['data']['train_processed'])
        X_train = train_data.drop('Churn', axis=1)
        
        # Make predictions
        predictions, probabilities = predict_churn(new_data, X_train)
        
        # Log request
        logging.info(f"Prediction request processed for {len(data)} customers at {datetime.now()}")
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist()
        )
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/mlflow/runs", response_model=List[MLflowRun])
async def get_mlflow_runs():
    """Retrieve MLflow experiment runs."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(config['mlflow']['experiment_name'])
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        run_data = []
        
        for run in runs:
            run_data.append(MLflowRun(
                run_id=run.info.run_id,
                start_time=datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                status=run.info.status,
                metrics=run.data.metrics,
                params=run.data.params
            ))
        
        logging.info(f"Retrieved {len(runs)} MLflow runs at {datetime.now()}")
        return run_data
    except Exception as e:
        logging.error(f"MLflow runs retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MLflow runs retrieval error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)