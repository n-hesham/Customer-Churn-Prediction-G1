# Customer Churn Prediction Project

## Overview
This project implements a machine learning pipeline to predict customer churn using a LightGBM model. It includes data ingestion from a MySQL database, preprocessing, model training, evaluation, prediction, and an API for serving predictions. MLflow is integrated for experiment tracking, and FastAPI provides a RESTful interface for predictions and experiment retrieval.

## Project Structure
```
Customer-Churn-Prediction-G1/
├── README.markdown              # Project documentation
├── requirements.txt            # Python dependencies
├── test_db.py                  # Database testing script
├── config/
│   └── config.yaml             # Configuration file (data paths, model settings, MLflow)
├── data/
│   ├── processed/              # Processed datasets
│   │   ├── processed_data.csv
│   │   ├── test_processed.csv
│   │   └── train_processed.csv
│   └── raw/
│       └── customer_churn_dataset-master_clean.csv  # Raw dataset
├── logs/
│   └── pipeline.log            # Pipeline execution logs
├── mlruns/                     # MLflow experiment tracking data
├── models/
│   ├── saved_preprocessing/    # Preprocessing artifacts
│   │   ├── feature_engineering_stats.pkl
│   │   ├── One_Hot_Encoder.pkl
│   │   ├── standard_scaler.pkl
│   │   └── stats.pkl
│   └── trained_models/
│       └── best_lgb_model.pkl  # Trained LightGBM model
├── notebooks/
│   ├── churn_prediction.ipynb  # Exploratory data analysis notebook
│   └── visuals/                # Visualizations from notebook
├── queries/
│   ├── age_distribution_binned.sql
│   ├── avg_metrics_by_churn.sql
│   ├── churn_by_contract_length.sql
│   ├── churn_by_usage_frequency.sql
│   ├── overview_count.sql
│   ├── subscription_type_churn.sql
│   ├── support_calls_vs_churn.sql
│   ├── tenure_distribution.sql
│   ├── total_spend_distribution.sql
│   └── usage_vs_churn.sql
├── src/
│   ├── api/
│   │   └── api.py              # FastAPI application for predictions and MLflow runs
│   ├── pipelines/
│   │   ├── data_ingestion.py   # Loads data from MySQL
│   │   ├── evaluation.py       # Evaluates model with metrics and MLflow logging
│   │   ├── mlflow_setup.py     # Configures MLflow tracking
│   │   ├── prediction.py       # Generates predictions for new data
│   │   ├── preprocessing.py    # Preprocesses data
│   │   ├── run_pipeline.py     # Orchestrates the full pipeline
│   │   ├── training.py         # Trains LightGBM model with MLflow logging
│   │   └── __init__.py
│   └── utils/
│       ├── visualization.py     # Visualization utilities
│       └── __pycache__/
├── visuals/                    # Pipeline-generated visualizations
│   ├── confusion_matrix.png
│   └── (other visualization files)
└── predictions.csv             # Prediction outputs from pipeline
```

## Prerequisites
- Python 3.8+
- MySQL database with customer churn data
- MLflow tracking server (optional, defaults to local file-based tracking)

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd Customer-Churn-Prediction-G1
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include:
   - `pandas`, `lightgbm`, `scikit-learn`, `imblearn`, `seaborn`, `matplotlib`
   - `sqlalchemy`, `mysql-connector-python`, `pyyaml`, `joblib`
   - `mlflow`, `fastapi`, `uvicorn`

3. **Set Up MySQL**:
   - Ensure a MySQL database is running.
   - Update `config/config.yaml` with your database credentials:
     ```yaml
     database:
       user: your_username
       password: your_password
       host: localhost
       database: your_database
     ```

4. **Set Up MLflow**:
   - For local tracking, ensure `mlruns/` is writable.
   - For a remote MLflow server, start it:
     ```bash
     mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
     ```
   - Update `config/config.yaml` with the tracking URI:
     ```yaml
     mlflow:
       tracking_uri: file://C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/mlruns
       experiment_name: CustomerChurnPrediction
     ```

## Usage

### Running the Pipeline
The pipeline performs data ingestion, preprocessing, training, evaluation, and prediction.

1. **Run the Full Pipeline**:
   ```bash
   python src/pipelines/run_pipeline.py
   ```
   - Loads data from MySQL using `data_ingestion.py`.
   - Preprocesses data with `preprocessing.py` (binning, encoding, scaling, feature engineering).
   - Trains a LightGBM model with `training.py`.
   - Evaluates the model with `evaluation.py`, saving a confusion matrix to `visuals/`.
   - Generates predictions for sample data with `prediction.py`, saving to `predictions.csv`.
   - Logs metrics, parameters, and artifacts to MLflow.

2. **View MLflow Experiments**:
   ```bash
   mlflow ui --backend-store-uri file://C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/mlruns
   ```
   Open `http://localhost:5000` to view runs, metrics, and artifacts.

### Running the API
The FastAPI application serves predictions and MLflow run data.

1. **Start the API**:
   ```bash
   python src/api/api.py
   ```
   The API runs at `http://localhost:8000`. Access the Swagger UI at `http://localhost:8000/docs`.

2. **API Endpoints**:
   - **POST /predict**:
     - Input: JSON list of customer data (e.g., `Age`, `Gender`, `Tenure`, etc.).
     - Output: Predictions (`0` or `1`) and churn probabilities.
     - Example:
       ```bash
       curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '[
           {
               "Age": 45,
               "Gender": "Female",
               "Tenure": 24,
               "UsageFrequency": 15,
               "SupportCalls": 0,
               "PaymentDelay": 0,
               "SubscriptionType": "Premium",
               "ContractLength": "Annual",
               "TotalSpend": 600.0,
               "LastInteraction": 30
           }
       ]'
       ```
   - **GET /mlflow/runs**:
     - Output: List of MLflow runs with run IDs, start times, status, metrics, and parameters.
     - Example:
       ```bash
       curl http://localhost:8000/mlflow/runs
       ```

### Configuration
Edit `config/config.yaml` to adjust:
- Database connection (`database` section).
- Data paths (`data` section, e.g., `train_processed`, `test_processed`).
- Model path (`model` section, e.g., `model_path`).
- Preprocessing artifact paths (`preprocessing` section, e.g., `encoder_path`).
- MLflow settings (`mlflow` section, e.g., `tracking_uri`).

## Pipeline Details
- **Data Ingestion**: Loads data from MySQL using a query defined in `config.yaml`.
- **Preprocessing**:
  - Bins `Age` and `Tenure` into groups.
  - Creates features like `Log_Usage_Rate`, `Spend_Per_Usage`, `High_Value`, and `At_Risk`.
  - Applies one-hot encoding to categorical columns and standard scaling to numerical columns.
  - Handles class imbalance with `RandomUnderSampler`.
  - Saves processed data to `data/processed/`.
- **Training**: Trains a LightGBM model with parameters logged to MLflow.
- **Evaluation**: Computes metrics (AUC, precision, recall, etc.) and logs them to MLflow. Saves a confusion matrix to `visuals/`.
- **Prediction**: Processes new data and returns churn predictions and probabilities.
- **MLflow**: Tracks experiments, logging parameters, metrics, models, and artifacts (e.g., confusion matrix, predictions).
- **API**: Provides endpoints for predictions and MLflow run retrieval.

## Logging
- All pipeline and API logs are saved to `logs/pipeline.log`.
- MLflow artifacts (e.g., models, confusion matrix) are stored in `mlruns/`.

## Visualizations
- SQL-based visualizations are generated in `notebooks/visuals/` and `visuals/` based on queries in `queries/`.
- The confusion matrix is saved to `visuals/confusion_matrix.png`.

## Testing
- Use `test_db.py` to verify MySQL connectivity.
- Test the API using the Swagger UI or tools like Postman or `curl`.

## Notes
- Ensure the MySQL database is accessible and populated with the required schema.
- For production API deployment, consider adding authentication and rate limiting.
- If using a remote MLflow server, update the `tracking_uri` in `config.yaml` and ensure the server is running.
- The pipeline assumes the model and preprocessing artifacts are available in `models/` and `models/saved_preprocessing/`.

## Contact
For issues or contributions, contact the project maintainer or submit a pull request.