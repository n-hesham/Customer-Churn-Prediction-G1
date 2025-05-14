import logging
import pandas as pd
import numpy as np # For select_dtypes
import mlflow
import os
from sklearn.metrics import roc_auc_score, classification_report

from mlops.pipeline import config 
from mlops.pipeline.utils import generate_data_drift_report, generate_data_quality_report

# Setup logging if not already configured by another entry point
if not logging.getLogger().hasHandlers() or all(isinstance(h, logging.NullHandler) for h in logging.getLogger().handlers):
    config.setup_logging() # Use the project's standard logging setup
logger = logging.getLogger(__name__)

# --- Configuration for Monitoring ---
LIVE_AUC_THRESHOLD = 0.70 # Alert if live AUC drops below this
MAX_DRIFTED_FEATURES_THRESHOLD = 3 # Alert if more than this many features drift


def get_production_data_sample_for_drift():
    """
    Placeholder: Fetches a recent sample of input features sent to the prediction service.
    This data should be *before* any scikit-learn transformations (OHE, scaling)
    but *after* your manual feature engineering if that's what your model pipeline expects.
    The structure should match X_train_fe / X_test_fe.
    """
    logger.info("Fetching recent production input data sample for drift analysis (placeholder)...")
    data = {
        'Age': [25, 45, 60, 33, 50, 22, 40, 55, 30, 65],
        'Tenure': [10, 20, 30, 5, 15, 2, 25, 35, 8, 40],
        'Usage Frequency': [15, 5, 20, 25, 10, 30, 12, 18, 22, 8],
        'Support Calls': [1, 0, 2, 0, 1, 3, 0, 1, 2, 0],
        'Payment Delay': [5, 0, 10, 2, 3, 0, 1, 0, 7, 4],
        'Total Spend': [100, 500, 800, 150, 600, 80, 700, 900, 200, 1000],
        'Last Interaction': [30, 90, 10, 180, 60, 5, 120, 20, 45, 150],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Subscription Type': ['Basic', 'Premium', 'Basic', 'Premium', 'Basic', 'Premium', 'Basic', 'Premium', 'Basic', 'Premium'],
        'Contract Length': ['Monthly', 'Annual', 'Monthly', 'Annual', 'Monthly', 'Annual', 'Monthly', 'Annual', 'Monthly', 'Annual'],
        # Manually engineered features would also be here if they are input to the sklearn pipeline
        'Log_Usage_Rate': np.random.rand(10) * 2,
        'Spend_Per_Usage': np.random.rand(10) * 100,
        'Payment_Delay_Ratio': np.random.rand(10),
        'High_Value': np.random.randint(0,2,10),
        'At_Risk': np.random.randint(0,2,10),
        'Age_group': ['Young', 'Adult', 'Senior', 'Adult', 'Senior', 'Young', 'Adult', 'Senior', 'Adult', 'Senior'],
        'tenure_group': ['<1yr', '1-2yr', '2-3yr', '<1yr', '1-2yr', '<1yr', '2-3yr', '3-4yr', '<1yr', '4-5yr']
    }
    # Ensure columns match `pipeline_input_columns` from fe_stats
    fe_stats = mlflow.artifacts.load_dict(os.path.join(mlflow.get_artifact_uri(config.FULL_PIPELINE_MODEL_PATH.replace(".pkl", "")),"fe_stats.json")) if os.path.exists(os.path.join(mlflow.get_artifact_uri(config.FULL_PIPELINE_MODEL_PATH.replace(".pkl", "")),"fe_stats.json")) else joblib.load(config.STATS_PATH) # Fallback to local
    expected_cols = fe_stats.get('pipeline_input_columns')
    
    prod_df = pd.DataFrame(data)
    if expected_cols:
        for col in expected_cols:
            if col not in prod_df.columns:
                logger.warning(f"Drift check: Expected column '{col}' not in production sample. Adding placeholder.")
                # Basic default logic
                if prod_df[expected_cols[0]].dtype == 'object' or isinstance(prod_df[expected_cols[0]].dtype, pd.CategoricalDtype): # Hacky way to guess type
                    prod_df[col] = "Unknown"
                else:
                    prod_df[col] = 0
        prod_df = prod_df[expected_cols]

    return prod_df

def get_reference_data_for_drift():
    """
    Placeholder: Fetches the reference dataset for drift comparison.
    This should be the X_train_fe data (after manual FE, before sklearn pipeline)
    from the training run of the current production model.
    """
    logger.info("Fetching reference data for drift analysis (placeholder)...")
    ref_df = get_production_data_sample_for_drift()
    logger.info(f"Using mocked reference data with shape: {ref_df.shape}")
    return ref_df


def get_recent_predictions_with_ground_truth():
    """
    Placeholder: Fetches recent predictions made by the production model
    and their corresponding ground truth (actual churn outcome).
    """
    logger.info("Fetching recent predictions and ground truth for performance monitoring (placeholder)...")
    data = {
        'request_id': [f'req_{i}' for i in range(50)],
        'prediction_label': np.random.randint(0, 2, 50),
        'prediction_proba_churn': np.random.rand(50),
        'actual_churn': np.random.randint(0, 2, 50) 
    }
    preds_df = pd.DataFrame(data)
    if preds_df.empty:
        logger.warning("No recent prediction data with ground truth found.")
        return None
    return preds_df


def monitor_model_performance(performance_df):
    if performance_df is None or performance_df.empty:
        logger.info("Skipping performance monitoring due to no data with ground truth.")
        return False # Indicates no monitoring done or issues

    logger.info(f"Monitoring model performance on {len(performance_df)} samples...")
    y_true = performance_df['actual_churn']
    y_pred_proba = performance_df['prediction_proba_churn']
    y_pred_label = performance_df['prediction_label']

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
        report_dict = classification_report(y_true, y_pred_label, output_dict=True, zero_division=0)
        accuracy = report_dict.get('accuracy', 0)
        f1_class1 = report_dict.get('1', {}).get('f1-score', 0) # F1 for churn class

        # Log to a dedicated monitoring experiment in MLflow
        monitor_experiment_name = config.MLFLOW_EXPERIMENT_NAME + "_Monitoring"
        try:
            experiment = mlflow.get_experiment_by_name(monitor_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(monitor_experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not get or create monitoring experiment '{monitor_experiment_name}': {e}")
            experiment_id = None # Fallback to default experiment if creation fails

        with mlflow.start_run(run_name="LivePerformanceCheck", experiment_id=experiment_id):
            mlflow.log_metric("live_auc", auc)
            mlflow.log_metric("live_accuracy", accuracy)
            mlflow.log_metric("live_f1_churn", f1_class1)
            mlflow.log_dict(report_dict, "live_classification_report.json")
            logger.info(f"Live Data Performance - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, F1 (Churn): {f1_class1:.4f}")

            if auc < LIVE_AUC_THRESHOLD:
                logger.error(f"ALERT: Live Model AUC ({auc:.4f}) is BELOW threshold ({LIVE_AUC_THRESHOLD})!")
                return True # Performance issue detected
            return False # Performance OK
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
        return True # Assume issue if metrics calculation fails


def monitor_data_drift_evidently(current_data_df, reference_data_df):
    if current_data_df is None or current_data_df.empty or reference_data_df is None or reference_data_df.empty:
        logger.info("Skipping data drift monitoring due to missing current or reference data.")
        return 0 # No drifted features detected / monitoring skipped

    logger.info(f"Monitoring data drift. Current data shape: {current_data_df.shape}, Reference data shape: {reference_data_df.shape}")
    
    # Determine numerical and categorical features from the data itself
    numerical_features = current_data_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = current_data_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Ensure reference data has the same columns in the same order
    try:
        reference_data_df = reference_data_df[current_data_df.columns]
    except KeyError as e:
        logger.error(f"Column mismatch between current and reference data for drift: {e}. Skipping drift.")
        missing_in_ref = set(current_data_df.columns) - set(reference_data_df.columns)
        missing_in_curr = set(reference_data_df.columns) - set(current_data_df.columns)
        logger.error(f"Missing in reference: {missing_in_ref}. Missing in current: {missing_in_curr}")
        return 0


    column_mapping_drift = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
    }

    drift_report_path = os.path.join(config.REPORTS_DIR, "live_data_drift_report.html")
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    generate_data_drift_report(
        reference_data=reference_data_df,
        current_data=current_data_df,
        report_path=drift_report_path,
        column_mapping=column_mapping_drift,
        profile_name="Live Data Input Drift"
    )
    
    # Placeholder: Logic to parse the Evidently report (HTML or JSON output)
    # to count drifted features. Evidently can output a JSON summary.
    # For now, a simple placeholder.
    # report = Report(metrics=[DataDriftPreset(num_stattest_threshold=0.05, cat_stattest_threshold=0.05)])
    # report.run(reference_data=reference_data_df, current_data=current_data_df, column_mapping=column_mapping_drift)
    # drift_summary = report.as_dict() # or report.as_json()
    # num_drifted_features_detected = drift_summary['metrics'][0]['result']['number_of_drifted_columns']
    num_drifted_features_detected = np.random.randint(0, 5) # Mocked
    logger.info(f"Detected {num_drifted_features_detected} drifted features (mocked value). Report at {drift_report_path}")


    monitor_experiment_name = config.MLFLOW_EXPERIMENT_NAME + "_Monitoring"
    try:
        experiment = mlflow.get_experiment_by_name(monitor_experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(monitor_experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
            logger.warning(f"Could not get or create monitoring experiment '{monitor_experiment_name}': {e}")
            experiment_id = None

    with mlflow.start_run(run_name="LiveDataDriftCheck", experiment_id=experiment_id):
        if os.path.exists(drift_report_path):
            mlflow.log_artifact(drift_report_path, "drift_reports")
        mlflow.log_metric("num_drifted_features", num_drifted_features_detected)

        if num_drifted_features_detected > MAX_DRIFTED_FEATURES_THRESHOLD:
            logger.warning(f"ALERT: {num_drifted_features_detected} features have drifted, exceeding threshold of {MAX_DRIFTED_FEATURES_THRESHOLD}!")
            return num_drifted_features_detected
        return num_drifted_features_detected


def main_monitoring_logic():
    logger.info("--- Starting Model Monitoring Pipeline ---")

    # Set MLflow tracking URI
    if hasattr(config, 'MLFLOW_TRACKING_URI') and config.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # 1. Monitor Model Performance
    live_predictions_with_gt = get_recent_predictions_with_ground_truth()
    performance_issue_detected = monitor_model_performance(live_predictions_with_gt)

    # 2. Monitor Data Drift on new input data
    production_input_sample = get_production_data_sample_for_drift()
    reference_data_sample = get_reference_data_for_drift()
    num_drifted = monitor_data_drift_evidently(production_input_sample, reference_data_sample)
    data_drift_issue_detected = num_drifted > MAX_DRIFTED_FEATURES_THRESHOLD

    # 3. (Optional) Trigger Retraining based on monitoring results
    if performance_issue_detected or data_drift_issue_detected:
        logger.warning("Monitoring detected issues. Suggesting retraining.")
        # Here you could call retrain_trigger.py or send a notification
        # For example:
        # from .retrain_trigger import main_retrain_logic
        # logger.info("Attempting to trigger retraining due to monitoring alerts...")
        # main_retrain_logic(force_retrain=True) # Force retrain based on alert
    else:
        logger.info("Monitoring checks passed. No immediate retraining trigger from monitoring.")

    logger.info("--- Model Monitoring Pipeline Finished ---")


if __name__ == "__main__":
    main_monitoring_logic()

    # Example usage:
    # python -m src.monitor_pipeline