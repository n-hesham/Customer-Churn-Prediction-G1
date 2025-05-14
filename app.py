from flask import Flask, request, jsonify
import joblib
import pandas as pd
import mlflow
import numpy as np
import logging
import os

try:
    from mlops.pipeline import config
    from mlops.pipeline.feature_engineering import _manual_feature_engineering_core
except ImportError:

    import config # Assumes config.py is in the same directory as app.py
    from feature_engineering import _manual_feature_engineering_core # Assumes feature_engineering.py is also accessible

app = Flask(__name__)

MLFLOW_MODEL_STAGE = "Production"
API_LOG_FILE = os.path.join(config.BASE_DIR, 'app_predictions.log') # Use BASE_DIR from config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # Log to console
        logging.FileHandler(API_LOG_FILE) # Log to file
    ]
)
logger = logging.getLogger(__name__)

pipeline_model = None
stats_loaded_globally = None

try:
    # Set MLflow tracking URI before any MLflow operations
    if hasattr(config, 'MLFLOW_TRACKING_URI') and config.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        logger.info(f"Set MLflow tracking URI to: {config.MLFLOW_TRACKING_URI}")
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        logger.info(f"Set MLflow tracking URI from environment: {os.getenv('MLFLOW_TRACKING_URI')}")
    else:
        logger.warning("MLFLOW_TRACKING_URI not set in config or environment. MLflow will use default local tracking.")

    model_uri = f"models:/{config.MLFLOW_MODEL_REGISTRY_NAME}/{MLFLOW_MODEL_STAGE}"
    logger.info(f"Attempting to load model from MLflow Registry: {model_uri}")
    pipeline_model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Model '{config.MLFLOW_MODEL_REGISTRY_NAME}/{MLFLOW_MODEL_STAGE}' loaded successfully from MLflow Registry.")

    logger.info(f"Attempting to load feature engineering stats from: {config.STATS_PATH}")
    stats_loaded_globally = joblib.load(config.STATS_PATH)
    logger.info("Feature engineering stats loaded successfully.")

except Exception as e:
    logger.error(f"CRITICAL ERROR during startup: Failed to load model or stats: {e}", exc_info=True)
    # For now, it will allow startup but predict endpoint will fail.
    pipeline_model = None
    stats_loaded_globally = None


def prepare_data_for_prediction_api(df_raw: pd.DataFrame, fe_stats_dict: dict):
    """
    Prepares new data for the prediction pipeline by applying manual feature engineering
    and aligning columns based on saved stats.
    This function mirrors the logic in `predict_script.py`'s `prepare_data_for_prediction`.
    """
    logger.info("Preparing data for API prediction...")
    df_to_process = df_raw.copy()

    # Drop CustomerID if present, as it's not used in FE or model
    if 'CustomerID' in df_to_process.columns:
        df_to_process = df_to_process.drop('CustomerID', axis=1)
        logger.info("Dropped 'CustomerID' column from prediction input.")

    # Apply manual feature engineering (transform mode)
    df_fe, _ = _manual_feature_engineering_core(
        df_to_process,
        stats_dict_for_creation=fe_stats_dict,
        fit_mode=False
    )
    logger.info(f"Manual FE complete for API prediction. Shape before alignment: {df_fe.shape}")

    pipeline_input_cols = fe_stats_dict.get('pipeline_input_columns')
    if not pipeline_input_cols:
        logger.error("'pipeline_input_columns' not found in fe_stats. This is critical for prediction.")
        raise ValueError("'pipeline_input_columns' missing from feature engineering stats.")

    # Align columns: Add missing (with defaults) and reorder to match pipeline_input_cols
    for col in pipeline_input_cols:
        if col not in df_fe.columns:
            logger.warning(f"API Prediction: Column '{col}' missing after manual FE. Adding with default.")
            if col in ['Age_group', 'tenure_group', 'Gender', 'Subscription Type', 'Contract Length']: # Categorical known from FE
                 df_fe[col] = "Unknown"
            else: # Assume numerical
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
    logger.debug(f"Columns for pipeline: {df_fe_aligned.columns.tolist()}")
    logger.debug(f"Data head for pipeline:\n{df_fe_aligned.head().to_string()}")
    return df_fe_aligned


def make_prediction_with_loaded_artifacts(new_data_df: pd.DataFrame):
    """
    Predicts churn using globally loaded model and stats.
    Applies feature engineering and column alignment.
    """
    if pipeline_model is None or stats_loaded_globally is None:
        logger.error("Model or stats not loaded. Cannot make predictions.")
        raise RuntimeError("Model or feature engineering stats are not loaded. Check server logs.")

    data_ready_for_pipeline = prepare_data_for_prediction_api(new_data_df, stats_loaded_globally)

    logger.info("Making predictions with the loaded pipeline model...")
    predictions = pipeline_model.predict(data_ready_for_pipeline)

    # Check if predict_proba is available (mlflow.pyfunc models often wrap sklearn models)
    # For scikit-learn models, predict_proba is usually available if the underlying model supports it.
    probabilities = None
    if hasattr(pipeline_model, 'predict_proba'):
        try:
            probabilities_all_classes = pipeline_model.predict_proba(data_ready_for_pipeline)
            # Assuming binary classification and we need probability of the positive class (usually index 1)
            if probabilities_all_classes.ndim == 2 and probabilities_all_classes.shape[1] >= 2:
                probabilities = probabilities_all_classes[:, 1]
            else: # Handle cases like single class output or unexpected shape
                logger.warning(f"predict_proba output shape {probabilities_all_classes.shape} not as expected for binary classification. Storing raw output.")
                probabilities = probabilities_all_classes.tolist() # Store as is if not 2D array of probs
        except Exception as e_proba:
            logger.warning(f"Could not get probabilities using predict_proba: {e_proba}. Probabilities will be null.")
    else: # Fallback if predict_proba is not directly available on the pyfunc model
        logger.warning("Model does not have predict_proba method. Probabilities will be null.")


    return predictions, probabilities, data_ready_for_pipeline


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if not pipeline_model or not stats_loaded_globally:
        logger.error("Prediction attempt when model/stats not loaded globally.")
        return jsonify({"error": "Model or feature engineering stats not loaded properly. Check server logs."}), 500

    try:
        json_data = request.get_json()
        if not json_data:
            logger.warning("No input data provided in request.")
            return jsonify({"error": "No input data provided"}), 400

        if isinstance(json_data, dict):
            data_for_df = [json_data]
        elif isinstance(json_data, list):
            data_for_df = json_data
        else:
            logger.warning(f"Invalid JSON payload format. Type: {type(json_data)}")
            return jsonify({"error": "Invalid JSON payload. Expected a JSON object or an array of JSON objects."}), 400

        if not data_for_df:
            logger.warning("Input data list is empty after parsing JSON.")
            return jsonify({"error": "Input data is empty"}), 400

        input_df = pd.DataFrame(data_for_df)
        logger.info(f"Received input data for prediction (shape: {input_df.shape}): \n{input_df.head().to_string()}")

        predictions, probabilities, _ = make_prediction_with_loaded_artifacts(input_df)

        response = {
            'predictions': predictions.tolist() if predictions is not None else None,
            'probabilities_churn': probabilities.tolist() if probabilities is not None and isinstance(probabilities, np.ndarray) else probabilities
        }
        logger.info(f"Prediction response: {response}")
        return jsonify(response)

    except ValueError as ve:
        logger.error(f"ValueError during prediction: {ve}", exc_info=True)
        return jsonify({"error": str(ve)}), 400
    except KeyError as ke:
        logger.error(f"KeyError during prediction: {ke} - likely a column mismatch.", exc_info=True)
        return jsonify({"error": f"Column error: {str(ke)}. Ensure input data and model expectations align."}), 400
    except RuntimeError as rt_err: # Catch the RuntimeError from make_prediction_with_loaded_artifacts
        logger.error(f"RuntimeError during prediction: {rt_err}", exc_info=True)
        return jsonify({"error": str(rt_err)}), 500
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred. Check server logs."}), 500

if __name__ == '__main__':
    # This check for MLflow experiment is more relevant for training pipelines,
    # but kept here for consistency if app.py were to log something to MLflow.
    # For a pure serving app, it might not be necessary.
    try:
        # Ensure tracking URI is set before this MLflow call too
        if hasattr(config, 'MLFLOW_TRACKING_URI') and config.MLFLOW_TRACKING_URI:
             mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        elif os.getenv("MLFLOW_TRACKING_URI"):
             mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # Use the experiment name from config.py
        if not mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME):
            mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)
            logger.info(f"MLflow experiment '{config.MLFLOW_EXPERIMENT_NAME}' created (or ensured).")
        else:
            logger.info(f"MLflow experiment '{config.MLFLOW_EXPERIMENT_NAME}' already exists.")
    except Exception as e_mlflow_init:
        logger.warning(f"Could not create or check MLflow experiment '{config.MLFLOW_EXPERIMENT_NAME}': {e_mlflow_init}")
        # App should still run even if MLflow experiment setup fails, as serving is primary.

    if pipeline_model and stats_loaded_globally:
        logger.info("Flask app starting: Model and stats loaded successfully.")
    else:
        logger.error("Flask app starting: CRITICAL - Model or stats FAILED to load. Predictions will fail.")

    app.run(debug=False, host='0.0.0.0', port=5001) # Set debug=False for production-like behavior