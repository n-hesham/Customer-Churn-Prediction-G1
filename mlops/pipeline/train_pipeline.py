import logging
import pandas as pd
import numpy
import os
import mlflow # Import mlflow here

# Use relative imports for modules within the same package (src)
from . import config
from . import data_loader
from . import feature_engineering
from . import model_trainer
# Import utils if you have it for data validation reports
from .utils import generate_data_quality_report, generate_data_drift_report

# Call setup_logging once at the beginning of the main script
config.setup_logging()
logger = logging.getLogger(__name__) # Get logger for this specific file

def run_training_pipeline():
    """
    Orchestrates the full model training pipeline.
    """
    logger.info("--- Churn Prediction Training Pipeline (Refactored & MLOps Integrated) Started ---")

    # Set MLflow tracking URI if defined in config or environment
    if hasattr(config, 'MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    else:
        logger.info("MLFLOW_TRACKING_URI not set, using default local tracking (mlruns folder).")

    run_managed_locally = False # Initialize flag
    if not mlflow.active_run():
        logger.info("No active MLflow run, starting a new one for train_pipeline.")
        experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            logger.info(f"Experiment '{config.MLFLOW_EXPERIMENT_NAME}' not found. Creating it.")
            experiment_id = mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.start_run(experiment_id=experiment_id, run_name="TrainPipelineRun_Manual")
        run_managed_locally = True
    else:
        logger.info(f"Already in an active MLflow run (ID: {mlflow.active_run().info.run_id}).")

    try:
        # 1. Load and split raw data
        try:
            X_train_raw, X_test_raw, y_train, y_test = data_loader.load_and_split_data(
                train_file_path=config.TRAIN_DATA_PATH,
                test_file_path=config.TEST_DATA_PATH
            )
            logger.info(f"Raw data loaded. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

            if config.GENERATE_DATA_REPORTS:
                logger.info("Generating initial data quality report for raw training data...")
                temp_train_raw_with_target = X_train_raw.copy()
                temp_train_raw_with_target[y_train.name if y_train.name else 'Churn'] = y_train
                
                generate_data_quality_report( # This will now print a warning and skip report generation
                    current_data=temp_train_raw_with_target,
                    report_path=config.DATA_QUALITY_REPORT_PATH,
                    profile_name="Initial Raw Training Data Quality"
                )
                # logger.info(f"Initial data quality report saved to {config.DATA_QUALITY_REPORT_PATH}") # This log is misleading if report is not generated
                # mlflow.log_artifact(config.DATA_QUALITY_REPORT_PATH, "data_validation_reports") # <-- TEMPORARILY COMMENTED OUT
                logger.info("Skipping MLflow logging for data quality report as it's currently disabled.")


        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}", exc_info=True)
            raise
        except ValueError as e:
            logger.error(f"ValueError during data loading: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"CRITICAL FAILURE at data loading: {e}", exc_info=True)
            raise

        # 2. Apply manual feature engineering
        try:
            X_train_fe, X_test_fe, calculated_fe_stats = feature_engineering.apply_manual_feature_engineering(
                X_train_raw, X_test_raw, stats_save_path=config.STATS_PATH
            )
            logger.info("Manual feature engineering complete.")
        except Exception as e:
            logger.error(f"CRITICAL FAILURE at manual FE: {e}", exc_info=True)
            raise

        # 3. Build, train, and log the scikit-learn model pipeline
        try:
            trained_sklearn_pipeline = model_trainer.build_and_train_model(
                X_train_fe, X_test_fe, y_train, y_test,
                fe_stats=calculated_fe_stats,
                model_save_path=config.FULL_PIPELINE_MODEL_PATH,
                mlflow_experiment_name=config.MLFLOW_EXPERIMENT_NAME,
                mlflow_model_registry_name=config.MLFLOW_MODEL_REGISTRY_NAME
            )
            if trained_sklearn_pipeline:
                logger.info("--- SKLearn Pipeline Training & MLflow logging/registration complete. ---")
            else:
                logger.error("--- SKLearn Pipeline Training FAILED (model not returned). ---")
                raise RuntimeError("Model training failed to return a pipeline.")
        except Exception as e:
            logger.error(f"CRITICAL FAILURE at model training: {e}", exc_info=True)
            raise

        # 4. Save processed datasets
        logger.info("Preparing to save processed datasets (after all scikit-learn transformations)...")
        try:
            preprocessor_from_pipeline = trained_sklearn_pipeline.named_steps.get('preprocessor')
            if not preprocessor_from_pipeline:
                logger.error("Preprocessor step not found in pipeline.")
                raise ValueError("Preprocessor step missing.")

            X_train_processed_np = preprocessor_from_pipeline.transform(X_train_fe)
            X_test_processed_np = preprocessor_from_pipeline.transform(X_test_fe)
            feature_names_out = preprocessor_from_pipeline.get_feature_names_out()

            X_train_processed_df = pd.DataFrame(X_train_processed_np, columns=feature_names_out, index=X_train_fe.index)
            X_test_processed_df = pd.DataFrame(X_test_processed_np, columns=feature_names_out, index=X_test_fe.index)

            train_processed_to_save = pd.concat([X_train_processed_df.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
            test_processed_to_save = pd.concat([X_test_processed_df.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

            train_processed_to_save.to_csv(config.TRAIN_PROCESSED_PATH, index=False)
            test_processed_to_save.to_csv(config.TEST_PROCESSED_PATH, index=False)
            logger.info(f"Processed training data saved to: {config.TRAIN_PROCESSED_PATH}")
            logger.info(f"Processed test data saved to: {config.TEST_PROCESSED_PATH}")

            if config.GENERATE_DATA_REPORTS:
                logger.info("Generating data drift report for processed data...")
                num_features_drift = list(feature_names_out)
                column_mapping_drift = {
                    'target': y_train.name if y_train.name else 'Churn',
                    'numerical_features': num_features_drift,
                    'categorical_features': []
                }
                generate_data_drift_report( # This will now print a warning and skip report generation
                    reference_data=train_processed_to_save,
                    current_data=test_processed_to_save,
                    report_path=config.PROCESSED_DRIFT_REPORT_PATH,
                    profile_name="Processed Train vs. Test Data Drift",
                    column_mapping=column_mapping_drift
                )
                # logger.info(f"Data drift report saved to {config.PROCESSED_DRIFT_REPORT_PATH}") # Misleading if not generated
                # mlflow.log_artifact(config.PROCESSED_DRIFT_REPORT_PATH, "data_validation_reports") # <-- TEMPORARILY COMMENTED OUT
                logger.info("Skipping MLflow logging for data drift report as it's currently disabled.")

        except Exception as e:
            logger.error(f"Error saving processed datasets or generating drift report: {e}", exc_info=True)
            # Non-critical, model is trained.

        logger.info("--- Churn Prediction Training Pipeline Finished Successfully. ---")
        if run_managed_locally:
            mlflow.end_run(status="FINISHED")

    except Exception as pipeline_error:
        logger.error(f"--- Churn Prediction Training Pipeline FAILED: {pipeline_error} ---", exc_info=True)
        if run_managed_locally and mlflow.active_run():
            mlflow.end_run(status="FAILED")
        # raise # Optionally re-raise if you want the script to exit with a non-zero code on failure


if __name__ == "__main__":
    run_training_pipeline()