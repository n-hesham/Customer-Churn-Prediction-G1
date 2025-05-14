import pandas as pd
import mlflow
import mlflow.sklearn # Explicitly import for log_model, infer_signature
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import logging
from matplotlib import pyplot as plt
import seaborn as sns
import os # For MLFLOW_TRACKING_URI
import numpy 

# Import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler


try:
    from . import config
except ImportError:
    import config


logger = logging.getLogger(__name__)

def build_and_train_model(
    X_train_fe: pd.DataFrame,
    X_test_fe: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fe_stats: dict,
    model_save_path: str = config.FULL_PIPELINE_MODEL_PATH,
    mlflow_experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
    mlflow_model_registry_name: str = config.MLFLOW_MODEL_REGISTRY_NAME # Get from config
):
    """
    Builds a scikit-learn pipeline with preprocessing (OneHotEncoder, StandardScaler)
    and a HistGradientBoostingClassifier. Trains the pipeline.
    If an active MLflow run exists, logs metrics, parameters, model, and registers the model.
    Saves the full pipeline locally.

    Args:
        X_train_fe: DataFrame of training features *after* manual feature engineering.
        X_test_fe: DataFrame of test features *after* manual feature engineering and alignment.
        y_train: Series of training target.
        y_test: Series of test target.
        fe_stats: Dictionary containing feature engineering statistics,
                  including 'pipeline_input_columns'. It might be updated with
                  cat/num cols used by the sklearn preprocessor.
        model_save_path: Path to save the trained scikit-learn pipeline.
        mlflow_experiment_name: Name of the MLflow experiment to use or create.
        mlflow_model_registry_name: Name to register the model under in MLflow Model Registry.
    """
    logger.info("Defining and training full preprocessing and classifier pipeline...")

    # --- START OF ADDED CODE ---
    # Handle missing values by dropping them from X_train_fe
    initial_rows_X_train_fe = len(X_train_fe)
    X_train_fe = X_train_fe.dropna()
    if len(X_train_fe) < initial_rows_X_train_fe:
        dropped_rows_count = initial_rows_X_train_fe - len(X_train_fe)
        logger.info(f"Dropped {dropped_rows_count} rows from X_train_fe due to NaN values.")
    
    # Adjust y_train to match the new index after dropping missing values
    y_train = y_train.loc[X_train_fe.index]
    logger.info(f"Shape of X_train_fe after NaN drop: {X_train_fe.shape}, y_train: {y_train.shape}")

    # Apply undersampling to reduce the samples from the majority class
    logger.info("Applying RandomUnderSampler to the training data.")
    undersampler = RandomUnderSampler(random_state=42)
    X_train_fe, y_train = undersampler.fit_resample(X_train_fe, y_train)
    logger.info(f"Shape of X_train_fe after undersampling: {X_train_fe.shape}")
    logger.info(f"Shape of y_train after undersampling: {y_train.shape}")
    logger.info(f"Class distribution in y_train after undersampling: \n{y_train.value_counts(normalize=True)}")
    # --- END OF ADDED CODE ---

    pipeline_input_cols = fe_stats.get('pipeline_input_columns')
    if not pipeline_input_cols:
        logger.error("'pipeline_input_columns' not found in fe_stats. Cannot build scikit-learn pipeline.")
        raise ValueError("'pipeline_input_columns' missing from feature engineering stats dictionary.")

    try:
        # Ensure dataframes contain only and all the specified columns in the correct order
        # X_train_fe and y_train are now potentially undersampled
        X_train_to_pipe = X_train_fe[pipeline_input_cols].copy()
        X_test_to_pipe = X_test_fe[pipeline_input_cols].copy() # X_test_fe is not undersampled
    except KeyError as e:
        missing_cols_train = set(pipeline_input_cols) - set(X_train_fe.columns)
        missing_cols_test = set(pipeline_input_cols) - set(X_test_fe.columns)
        logger.error(f"KeyError when selecting pipeline_input_columns: {e}. ")
        if missing_cols_train:
             logger.error(f"Missing in X_train_fe: {missing_cols_train}")
        if missing_cols_test:
             logger.error(f"Missing in X_test_fe: {missing_cols_test}")
        logger.error(f"Expected columns: {pipeline_input_cols}")
        raise

    cat_cols_for_sklearn_pipeline = []
    num_cols_for_sklearn_pipeline = []

    for col in pipeline_input_cols:
        # Ensure the column exists in X_train_to_pipe before checking its dtype
        if col not in X_train_to_pipe.columns:
            logger.error(f"Pipeline input column '{col}' not found in X_train_to_pipe. This should not happen if alignment was correct.")
            raise KeyError(f"Configured pipeline input column '{col}' is missing from the training data provided to the scikit-learn pipeline.")

        col_dtype = X_train_to_pipe[col].dtype
        if pd.api.types.is_object_dtype(col_dtype) or \
           pd.api.types.is_string_dtype(col_dtype) or \
           pd.api.types.is_categorical_dtype(col_dtype):
            cat_cols_for_sklearn_pipeline.append(col)
        elif pd.api.types.is_numeric_dtype(col_dtype): # Corrected to use pd.api.types
            num_cols_for_sklearn_pipeline.append(col)
        else:
            logger.warning(f"Column '{col}' has an unhandled dtype '{col_dtype}' and will be passed through by ColumnTransformer if remainder='passthrough'.")

    logger.info(f"Categorical columns for SKLearn OneHotEncoder: {cat_cols_for_sklearn_pipeline}")
    logger.info(f"Numerical columns for SKLearn StandardScaler: {num_cols_for_sklearn_pipeline}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols_for_sklearn_pipeline),
            ('scaler', StandardScaler(), num_cols_for_sklearn_pipeline)
        ],
        remainder='passthrough' 
    )

    full_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(random_state=42, class_weight='balanced')) # Explicitly use LGBMClassifier from lgb
    ])

    logger.info(f"Fitting the full scikit-learn model pipeline on X_train_to_pipe (shape: {X_train_to_pipe.shape}) and y_train (shape: {y_train.shape})")
    try:
        full_model_pipeline.fit(X_train_to_pipe, y_train) # y_train is now the undersampled version
        logger.info("Full scikit-learn pipeline fitted successfully.")
    except Exception as e:
        logger.error(f"Error during fitting of the scikit-learn pipeline: {e}", exc_info=True)
        raise 

    active_mlflow_run = mlflow.active_run()
    if active_mlflow_run:
        logger.info(f"Active MLflow run detected (ID: {active_mlflow_run.info.run_id}). Proceeding with logging.")
        try:
            logger.info(f"Predicting on X_test_to_pipe (shape: {X_test_to_pipe.shape}) for metrics.")
            y_pred = full_model_pipeline.predict(X_test_to_pipe)
            y_pred_proba = full_model_pipeline.predict_proba(X_test_to_pipe)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba) # y_test is the original, not undersampled

            logger.info(f"Test AUC Score (Full Pipeline): {auc_score:.4f}")
            mlflow.log_metric("test_auc_score", auc_score)

            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            mlflow.log_metric("test_accuracy", report_dict['accuracy'])

            if '1' in report_dict and isinstance(report_dict['1'], dict): 
                mlflow.log_metric("test_precision_class1", report_dict['1'].get('precision', 0))
                mlflow.log_metric("test_recall_class1", report_dict['1'].get('recall', 0))
                mlflow.log_metric("test_f1_class1", report_dict['1'].get('f1-score', 0))
            elif '0' in report_dict and isinstance(report_dict['0'], dict): 
                logger.warning("Positive class '1' not found in classification report. Logging metrics for class '0'.")
                mlflow.log_metric("test_precision_class0", report_dict['0'].get('precision', 0))
                mlflow.log_metric("test_recall_class0", report_dict['0'].get('recall', 0))
                mlflow.log_metric("test_f1_class0", report_dict['0'].get('f1-score', 0))

            mlflow.log_dict(report_dict, "classification_report_test.json")

            logger.info("Classification Report (Test Set - Full Pipeline):\n" +
                        classification_report(y_test, y_pred, zero_division=0))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix (Test Set - Full Pipeline)')
            ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
            mlflow.log_figure(fig, "confusion_matrix_test.png")
            plt.close(fig) 

            mlflow.log_param("model_type", full_model_pipeline.named_steps['classifier'].__class__.__name__)
            classifier_params = full_model_pipeline.named_steps['classifier'].get_params()
            simple_classifier_params = {k: v for k, v in classifier_params.items() if isinstance(v, (str, int, float, bool, type(None)))}
            mlflow.log_params({f"classifier_{k}": v for k, v in simple_classifier_params.items()})
            
            # Log undersampling parameter
            mlflow.log_param("undersampling_method", "RandomUnderSampler")
            mlflow.log_param("undersampling_random_state", 42)


            mlflow.log_param("fe_pipeline_input_cols_count", len(pipeline_input_cols))
            mlflow.log_param("sklearn_cat_features_ohe_count", len(cat_cols_for_sklearn_pipeline))
            mlflow.log_param("sklearn_num_features_scaled_count", len(num_cols_for_sklearn_pipeline))

            input_example_df = X_train_to_pipe.head() if not X_train_to_pipe.empty else None
            signature = None
            if input_example_df is not None and not input_example_df.empty:
                try:
                    signature_predictions = full_model_pipeline.predict(input_example_df)
                    signature = mlflow.models.infer_signature(input_example_df, signature_predictions)
                except Exception as e_sig:
                    logger.warning(f"Could not infer signature for MLflow model: {e_sig}", exc_info=True)
            else:
                 logger.warning("Input example for MLflow signature is None or empty.")
            
            mlflow_artifact_path = "lgb-churn-full-sklearn-pipeline" 
            mlflow.sklearn.log_model(
                sk_model=full_model_pipeline,
                artifact_path=mlflow_artifact_path,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                input_example=input_example_df, 
                signature=signature 
            )
            logger.info(f"Full scikit-learn pipeline model logged to MLflow artifact path: '{mlflow_artifact_path}'.")

            if mlflow_model_registry_name:
                model_uri_for_registry = f"runs:/{active_mlflow_run.info.run_id}/{mlflow_artifact_path}"
                try:
                    mlflow.register_model(
                        model_uri=model_uri_for_registry,
                        name=mlflow_model_registry_name
                    )
                    logger.info(f"Model registered in MLflow Model Registry with name: '{mlflow_model_registry_name}' from run_id: {active_mlflow_run.info.run_id}")
                except Exception as e_reg:
                    logger.warning(f"Could not register model to MLflow Model Registry (name: '{mlflow_model_registry_name}'). "
                                   f"Error: {e_reg}", exc_info=True)
            else:
                logger.info("mlflow_model_registry_name not provided, skipping model registration to registry.")

        except Exception as e_mlflow:
            logger.error(f"Error during MLflow logging operations: {e_mlflow}", exc_info=True)
    else:
        logger.warning("No active MLflow run found. Skipping MLflow logging and model registration.")

    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True) 
        joblib.dump(full_model_pipeline, model_save_path)
        logger.info(f"Full scikit-learn pipeline model saved locally to {model_save_path}")
    except Exception as e_save:
        logger.error(f"Error saving scikit-learn pipeline model locally to {model_save_path}: {e_save}", exc_info=True)
        raise 

    return full_model_pipeline