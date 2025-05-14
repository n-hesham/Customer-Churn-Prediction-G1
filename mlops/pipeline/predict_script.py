import pandas as pd
import joblib
import mlflow
import argparse
import logging
import os

# Use relative import if predict_script.py is inside src
# If predict_script.py is in project root, it needs src in PYTHONPATH or direct pathing.
# Assuming it's run as 'python -m src.predict_script ...' or src is in PYTHONPATH
from . import config # Use relative import
from .feature_engineering import _manual_feature_engineering_core # Import the core FE function

# Setup logging if it's not already configured by calling script
# This check can be problematic if other modules already set up basicConfig with different settings.
# It's generally better to have one central setup_logging call.
# config.setup_logging() is called in train_pipeline.py. If predict_script is run standalone,
# it might need its own explicit call if not already handled.
if not logging.getLogger().hasHandlers() or all(isinstance(h, logging.NullHandler) for h in logging.getLogger().handlers):
    # A minimal config if not already set, could also call config.setup_logging()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def load_model_for_prediction(model_name_registry, model_stage_registry, local_model_path):
    """
    Tries to load a model from MLflow Model Registry first, then from a local path.
    The model loaded should be the full scikit-learn pipeline.
    """
    # MLflow tracking URI should be set before this call, typically at script start
    try:
        model_uri = f"models:/{model_name_registry}/{model_stage_registry}"
        logger.info(f"Attempting to load model from MLflow Registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri) # mlflow.pyfunc can load sklearn pipelines
        logger.info(f"Loaded model '{model_name_registry}' stage '{model_stage_registry}' from MLflow Registry.")
        return model
    except Exception as e_mlflow:
        logger.warning(f"Failed to load model from MLflow Registry (URI: {model_uri}): {e_mlflow}. "
                       f"Attempting to load from local path: {local_model_path}")
        try:
            model = joblib.load(local_model_path)
            logger.info(f"Successfully loaded model from local path: {local_model_path}")
            return model
        except Exception as e_local:
            logger.error(f"Failed to load model from local path '{local_model_path}': {e_local}", exc_info=True)
            raise ValueError(f"Could not load model from MLflow Registry or local path: {local_model_path}") from e_local


def prepare_data_for_prediction(df_raw, fe_stats_path):
    """
    Applies manual feature engineering to raw data using saved stats.
    Aligns columns to match the expected input of the scikit-learn pipeline.
    """
    logger.info(f"Loading feature engineering stats from: {fe_stats_path}")
    try:
        fe_stats = joblib.load(fe_stats_path)
    except FileNotFoundError:
        logger.error(f"Feature engineering stats file not found at {fe_stats_path}. Cannot proceed with prediction.")
        raise
    except Exception as e:
        logger.error(f"Error loading feature engineering stats: {e}", exc_info=True)
        raise

    logger.info("Applying manual feature engineering (transform mode)...")
    # Ensure CustomerID is dropped if present, as it's not used in FE or model
    df_to_process = df_raw.copy()
    if 'CustomerID' in df_to_process.columns:
        df_to_process = df_to_process.drop('CustomerID', axis=1)
        logger.info("Dropped 'CustomerID' column from prediction input.")

    df_fe, _ = _manual_feature_engineering_core(
        df_to_process,
        stats_dict_for_creation=fe_stats,
        fit_mode=False
    )
    logger.info(f"Manual FE complete for prediction. Shape after FE: {df_fe.shape}")

    pipeline_input_cols = fe_stats.get('pipeline_input_columns')
    if not pipeline_input_cols:
        logger.error("'pipeline_input_columns' not found in fe_stats. This is critical for prediction.")
        raise ValueError("'pipeline_input_columns' missing from feature engineering stats.")

    # Align columns: Add missing and reorder
    # This part should mirror the alignment logic from feature_engineering.py for test data
    for col in pipeline_input_cols:
        if col not in df_fe.columns:
            logger.warning(f"Prediction: Column '{col}' missing after manual FE. Adding with default.")
            # Determine default value. This logic needs to be robust.
            # Mirroring default logic from feature_engineering.py 'align columns' section
            # A placeholder: if 'Age_group' or 'tenure_group' use "Unknown", else 0.
            # Ideally, fe_stats would store dtypes or default fill values for each generated column.
            if col in ['Age_group', 'tenure_group', 'Gender', 'Subscription Type', 'Contract Length']: # these are known categorical from FE
                 df_fe[col] = "Unknown"
            else: # For other numerical or unknown types, default to 0 for simplicity
                 df_fe[col] = 0.0
            logger.info(f"Added missing column '{col}' with a default value.")


    try:
        df_fe_aligned = df_fe[pipeline_input_cols]
    except KeyError as e:
        missing_cols = set(pipeline_input_cols) - set(df_fe.columns)
        extra_cols = set(df_fe.columns) - set(pipeline_input_cols)
        logger.error(f"Column mismatch for prediction pipeline input. Missing: {missing_cols}, Extra: {extra_cols}", exc_info=True)
        raise KeyError(f"Error aligning columns for prediction: {e}. Check fe_stats and manual FE output.")

    logger.info(f"Data prepared and aligned for scikit-learn pipeline. Final shape: {df_fe_aligned.shape}")
    return df_fe_aligned


def make_predictions(input_data_path, output_predictions_path, model, fe_stats_path):
    logger.info(f"Loading raw data for prediction from: {input_data_path}")
    try:
        df_raw = pd.read_csv(input_data_path)
    except FileNotFoundError:
        logger.error(f"Input data file not found: {input_data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading input data CSV: {e}", exc_info=True)
        raise

    # Prepare data (apply manual FE and align columns)
    df_for_pipeline = prepare_data_for_prediction(df_raw, fe_stats_path)

    logger.info("Making predictions with the loaded model...")
    try:
        predictions_labels = model.predict(df_for_pipeline)
        # Check if predict_proba is available (it is for HistGradientBoostingClassifier and pyfunc-wrapped sklearn)
        if hasattr(model, 'predict_proba'):
            predictions_proba_all = model.predict_proba(df_for_pipeline)
            if predictions_proba_all.ndim == 2 and predictions_proba_all.shape[1] >=2:
                 predictions_proba = predictions_proba_all[:, 1] # Probability of class 1
            else:
                logger.warning(f"predict_proba output shape {predictions_proba_all.shape} not as expected. Storing raw probabilities.")
                predictions_proba = predictions_proba_all.tolist()
        else:
            logger.warning("Model does not have predict_proba method. Saving only labels.")
            predictions_proba = None
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        raise

    output_df = df_raw.copy() # Start with original raw data to append predictions
    if 'CustomerID' in df_raw.columns: # Keep CustomerID if it was in the input for easier joining later
        output_df = df_raw[['CustomerID']].copy()
    else: # If no CustomerID, just create an empty df to append to
        output_df = pd.DataFrame(index=df_raw.index)

    output_df['Predicted_Churn_Label'] = predictions_labels
    if predictions_proba is not None:
        output_df['Predicted_Churn_Probability'] = predictions_proba

    try:
        os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
        output_df.to_csv(output_predictions_path, index=False)
        logger.info(f"Predictions successfully saved to: {output_predictions_path}")
    except Exception as e:
        logger.error(f"Error saving predictions CSV: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Prediction Batch Inference Script")
    parser.add_argument("--input_data", required=True, help="Path to the input CSV data for prediction (raw format).")
    parser.add_argument("--output_preds", required=True, help="Path to save the predictions CSV.")
    parser.add_argument("--fe_stats_path", default=config.STATS_PATH, help="Path to feature engineering stats file.")
    parser.add_argument("--model_name", default=config.MLFLOW_MODEL_REGISTRY_NAME, help="Name of the model in MLflow Registry.") # Use from config
    parser.add_argument("--model_stage", default="Production", help="Stage of the model in MLflow Registry (e.g., Production, Staging).")
    parser.add_argument("--local_model_path", default=config.FULL_PIPELINE_MODEL_PATH, help="Fallback local path to the serialized model pipeline.")

    args = parser.parse_args()

    # Set MLflow tracking URI explicitly
    if hasattr(config, 'MLFLOW_TRACKING_URI') and config.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        logger.info(f"Set MLflow tracking URI to: {config.MLFLOW_TRACKING_URI}")
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        logger.info(f"Set MLflow tracking URI from environment: {os.getenv('MLFLOW_TRACKING_URI')}")
    else:
        logger.info("MLFLOW_TRACKING_URI not set, using default local tracking (mlruns folder).")


    # Ensure output directory for predictions exists
    os.makedirs(os.path.dirname(args.output_preds), exist_ok=True)

    loaded_model = load_model_for_prediction(args.model_name, args.model_stage, args.local_model_path)
    make_predictions(args.input_data, args.output_preds, loaded_model, args.fe_stats_path)

    logger.info("Batch prediction script finished successfully.")
    # Example usage from project root:
    # python -m src.predict_script --input_data data/raw/customer_churn_dataset-testing-master.csv --output_preds data/predictions/test_predictions.csv