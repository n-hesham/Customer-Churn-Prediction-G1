# src/config.py
import os
import logging

# --- Path Definitions ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Should point to project root

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

MODELS_DIR = os.path.join(BASE_DIR, "models")
PREPROCESSING_SAVE_DIR = os.path.join(MODELS_DIR, "saved_preprocessing")

# --- File Path Definitions ---
TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "customer_churn_dataset-training-master.csv")
TEST_DATA_PATH = os.path.join(RAW_DATA_DIR, "customer_churn_dataset-testing-master.csv")

STATS_PATH = os.path.join(PREPROCESSING_SAVE_DIR, "feature_engineering_stats.pkl")
FULL_PIPELINE_MODEL_PATH = os.path.join(MODELS_DIR, "best_lgb_model.pkl")

TRAIN_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv')
TEST_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv')

SAMPLE_PREDICTION_OUTPUT_PATH = os.path.join(PREDICTIONS_DIR, "sample_predictions.csv")
DATA_QUALITY_REPORT_PATH = os.path.join(REPORTS_DIR, "initial_train_raw_data_quality_report.html")
PROCESSED_DRIFT_REPORT_PATH = os.path.join(REPORTS_DIR, "processed_train_test_drift_report.html")

# --- MLflow Configuration ---
MLFLOW_EXPERIMENT_NAME = "Churn_Prediction_Refactored_Pipeline"
MLFLOW_MODEL_REGISTRY_NAME = "ChurnPredictionHGB_Refactored" # Name for Model Registry
# Optional: If using a remote tracking server, set this environment variable
# or uncomment and set here:
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# --- Data Validation Configuration ---
GENERATE_DATA_REPORTS = True # Set to False to disable Evidently report generation

# Create directories
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREPROCESSING_SAVE_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Logging Configuration ---
def setup_logging(level=logging.INFO):
    """Configures basic logging for the application."""
    # Check if logging is already configured to avoid adding multiple handlers.
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers() or \
       all(isinstance(h, logging.NullHandler) for h in root_logger.handlers):
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(), # Outputs to console
                # Optional: Add FileHandler for logging to a file
                # logging.FileHandler(os.path.join(BASE_DIR, "pipeline.log"))
            ]
        )
    # Set levels for noisy third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("seaborn").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING) # Often noisy
    # Add other libraries as needed