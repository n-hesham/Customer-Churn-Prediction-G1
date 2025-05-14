import logging
import subprocess # To call train_pipeline.py
import mlflow
import os
import sys

# Add src to path for imports if running script directly from project root for example
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlops.pipeline import config
from mlops.pipeline import utils
from mlops.pipeline.feature_engineering import _manual_feature_engineering_core
# Setup logging
if not logging.getLogger().hasHandlers() or all(isinstance(h, logging.NullHandler) for h in logging.getLogger().handlers):
    config.setup_logging()
logger = logging.getLogger(__name__)

# --- Configuration for Retraining Logic ---
# These thresholds would ideally come from a config or be more dynamic
PERFORMANCE_DROP_THRESHOLD_AUC = 0.65 # Example: If production AUC drops below this, trigger retrain
NEW_MODEL_IMPROVEMENT_THRESHOLD_AUC = 0.01 # New model must be this much better

def get_production_model_performance():
    """
    Fetches performance metrics of the current model in 'Production' stage.
    This is a placeholder. In reality, you'd fetch this from your monitoring system
    or by querying MLflow for the latest metrics of the production model on recent data.
    """
    logger.info("Fetching current production model performance (placeholder)...")
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(config.MLFLOW_MODEL_REGISTRY_NAME, stages=["Production"])
        if not versions:
            logger.warning("No model found in Production stage. Assuming retraining is needed.")
            return None # Or a very low value to force retrain

        prod_model_version = versions[0]
        run_info = client.get_run(prod_model_version.run_id)
        
        # Check if 'test_auc_score' metric exists for the production model's run
        if 'test_auc_score' in run_info.data.metrics:
            prod_auc = run_info.data.metrics['test_auc_score']
            logger.info(f"Production model (Version: {prod_model_version.version}, Run ID: {prod_model_version.run_id}) AUC: {prod_auc:.4f}")
            return prod_auc
        else:
            logger.warning(f"Metric 'test_auc_score' not found for production model run {prod_model_version.run_id}. Cannot assess performance.")
            return None # Cannot determine performance
    except Exception as e:
        logger.error(f"Error fetching production model performance: {e}", exc_info=True)
        return None


def run_training_pipeline_script():
    """
    Executes the main training pipeline script.
    """
    logger.info("Attempting to run the training pipeline (src.train_pipeline)...")
    try:
        # Assuming train_pipeline.py is executable and set up to be run as a module
        # The command depends on how your environment and PYTHONPATH are set up
        # If src is in PYTHONPATH: python -m src.train_pipeline
        # If running from project root: python src/train_pipeline.py
        process = subprocess.run(
            [sys.executable, "-m", "src.train_pipeline"], # More robust way to call python scripts
            capture_output=True, text=True, check=True
        )
        logger.info("Training pipeline script executed successfully.")
        logger.debug(f"Training script stdout:\n{process.stdout}")
        if process.stderr:
             logger.warning(f"Training script stderr:\n{process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training pipeline script failed with exit code {e.returncode}.")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Could not find python interpreter or train_pipeline.py. Check path and command.")
        return False


def compare_and_promote_model():
    """
    Compares the latest 'Staging' (or non-archived, non-production) model with the 'Production' model.
    Promotes the new model if it's significantly better.
    """
    logger.info("Comparing newly trained model with production model...")
    client = mlflow.tracking.MlflowClient()

    try:
        prod_versions = client.get_latest_versions(config.MLFLOW_MODEL_REGISTRY_NAME, stages=["Production"])
        # Get the latest version that is not 'Archived' and not 'Production'
        # This assumes new models are registered without a stage or in 'Staging'
        all_versions = client.get_latest_versions(config.MLFLOW_MODEL_REGISTRY_NAME, stages=None) # Get all, then filter
        
        candidate_model_version = None
        for v in sorted(all_versions, key=lambda x: int(x.version), reverse=True):
            if v.current_stage.lower() not in ["production", "archived"]:
                candidate_model_version = v
                break
        
        if not candidate_model_version:
            logger.warning("No new candidate model found for comparison (e.g., in 'Staging' or 'None' stage).")
            return False

        logger.info(f"Candidate model: Version {candidate_model_version.version}, Stage: {candidate_model_version.current_stage}, Run ID: {candidate_model_version.run_id}")

        candidate_run_info = client.get_run(candidate_model_version.run_id)
        if 'test_auc_score' not in candidate_run_info.data.metrics:
            logger.error(f"Metric 'test_auc_score' not found for candidate model run {candidate_model_version.run_id}. Cannot compare.")
            return False
        candidate_auc = candidate_run_info.data.metrics['test_auc_score']
        logger.info(f"Candidate model test_auc_score: {candidate_auc:.4f}")

        if not prod_versions:
            logger.info("No model currently in Production. Promoting candidate model.")
            client.transition_model_version_stage(
                name=config.MLFLOW_MODEL_REGISTRY_NAME,
                version=candidate_model_version.version,
                stage="Production",
                archive_existing_versions=True # Good practice
            )
            logger.info(f"Promoted candidate model Version {candidate_model_version.version} to Production.")
            return True

        prod_model_version = prod_versions[0]
        prod_run_info = client.get_run(prod_model_version.run_id)
        if 'test_auc_score' not in prod_run_info.data.metrics:
            logger.warning(f"Metric 'test_auc_score' not found for production model run {prod_model_version.run_id}. Promoting candidate.")
            # Decide policy: promote if prod metric missing, or hold? For now, promote.
            client.transition_model_version_stage(
                name=config.MLFLOW_MODEL_REGISTRY_NAME,
                version=candidate_model_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f"Promoted candidate model Version {candidate_model_version.version} to Production (production AUC was missing).")
            return True

        prod_auc = prod_run_info.data.metrics['test_auc_score']
        logger.info(f"Current Production model (Version {prod_model_version.version}) test_auc_score: {prod_auc:.4f}")

        if candidate_auc > prod_auc + NEW_MODEL_IMPROVEMENT_THRESHOLD_AUC:
            logger.info(f"Candidate model AUC ({candidate_auc:.4f}) is better than Production AUC ({prod_auc:.4f}) by more than threshold.")
            client.transition_model_version_stage(
                name=config.MLFLOW_MODEL_REGISTRY_NAME,
                version=candidate_model_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            # Optionally archive the old production model
            client.transition_model_version_stage(
                name=config.MLFLOW_MODEL_REGISTRY_NAME,
                version=prod_model_version.version, # Old production version
                stage="Archived"
            )
            logger.info(f"Promoted candidate model Version {candidate_model_version.version} to Production. Archived old Production version {prod_model_version.version}.")
            return True
        else:
            logger.info(f"Candidate model AUC ({candidate_auc:.4f}) is not sufficiently better than Production AUC ({prod_auc:.4f}). No promotion.")
            # Optionally, set candidate to Archived if not promoted
            client.transition_model_version_stage(
                name=config.MLFLOW_MODEL_REGISTRY_NAME,
                version=candidate_model_version.version,
                stage="Archived" # Or keep in 'Staging' for manual review
            )
            logger.info(f"Archived candidate model Version {candidate_model_version.version} as it was not promoted.")
            return False

    except Exception as e:
        logger.error(f"Error during model comparison and promotion: {e}", exc_info=True)
        return False


def main_retrain_logic(force_retrain=False):
    """
    Main logic for deciding whether to retrain and promote.
    """
    logger.info("--- Starting Retraining Trigger Script ---")

    # Set MLflow tracking URI
    if hasattr(config, 'MLFLOW_TRACKING_URI') and config.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    else:
        logger.info("MLFLOW_TRACKING_URI not set, using default local tracking.")


    perform_retrain = force_retrain
    if not force_retrain:
        production_auc = get_production_model_performance()
        if production_auc is None: # No prod model or performance unknown
            logger.info("Retraining due to no production model or unknown performance.")
            perform_retrain = True
        elif production_auc < PERFORMANCE_DROP_THRESHOLD_AUC:
            logger.info(f"Production model AUC ({production_auc:.4f}) is below threshold ({PERFORMANCE_DROP_THRESHOLD_AUC}). Triggering retraining.")
            perform_retrain = True
        else:
            logger.info(f"Production model AUC ({production_auc:.4f}) is above threshold. No performance-triggered retraining needed.")

    if perform_retrain:
        logger.info("Proceeding with model retraining...")
        training_successful = run_training_pipeline_script()
        if training_successful:
            logger.info("Training pipeline completed. Proceeding to model comparison and promotion.")
            compare_and_promote_model()
        else:
            logger.error("Training pipeline failed. Model comparison and promotion will be skipped.")
    else:
        logger.info("Retraining not triggered based on current conditions.")

    logger.info("--- Retraining Trigger Script Finished ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trigger model retraining and conditional promotion.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if performance thresholds are met."
    )
    args = parser.parse_args()

    main_retrain_logic(force_retrain=args.force)

    # Example usage:
    # python -m src.retrain_trigger
    # python -m src.retrain_trigger --force