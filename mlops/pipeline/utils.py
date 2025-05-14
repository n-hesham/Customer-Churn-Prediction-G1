import pandas as pd
import logging
import os

# --- Evidently AI Imports ---
# Ensure Evidently is installed: pip install evidently
try:
    from evidently.report import Report
    from evidently.metric_preset import DataQualityPreset, DataDriftPreset # For general drift
    # For more specific drift metrics if needed:
    # from evidently.metrics import DatasetDriftMetric, DataDriftTable, ColumnDriftMetric
    from evidently.pipeline.column_mapping import ColumnMapping # Correct import path
    EVIDENTLY_INSTALLED = True
except ImportError:
    EVIDENTLY_INSTALLED = False
    # Define dummy classes or functions if Evidently is not installed,
    # so the rest of the code doesn't break if it tries to use these types.
    class Report: pass
    class DataQualityPreset: pass
    class DataDriftPreset: pass
    class ColumnMapping: pass
    logging.getLogger(__name__).warning(
        "Evidently AI library not found. Data report generation will be skipped. "
        "Please install with 'pip install evidently'"
    )


logger = logging.getLogger(__name__)

def generate_data_quality_report(current_data: pd.DataFrame, report_path: str, profile_name: str = "Data Quality"):
    """
    Generates a data quality report using Evidently AI.
    """
    if not EVIDENTLY_INSTALLED:
        logger.warning(f"Evidently AI not installed. Skipping {profile_name} report generation.")
        return

    if current_data is None or current_data.empty:
        logger.warning(f"Input data for {profile_name} report is empty. Skipping report generation.")
        return

    logger.info(f"Generating {profile_name} report for data with shape {current_data.shape}...")
    try:
        # For DataQualityPreset, column_mapping is usually not strictly necessary
        # unless you have specific target/prediction columns you want to highlight,
        # but for general quality, it can often infer.
        data_quality_report = Report(metrics=[DataQualityPreset()])
        data_quality_report.run(current_data=current_data, reference_data=None, column_mapping=None)

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        data_quality_report.save_html(report_path)
        logger.info(f"{profile_name} report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to generate {profile_name} report: {e}", exc_info=True)


def generate_data_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                               report_path: str, profile_name: str = "Data Drift",
                               column_mapping: dict = None):
    """
    Generates a data drift report using Evidently AI.

    Args:
        reference_data: The baseline dataset (e.g., training data).
        current_data: The dataset to compare against the baseline (e.g., recent production data).
        report_path: Path to save the HTML report.
        profile_name: Name for logging purposes.
        column_mapping: A dictionary specifying column types for Evidently.
                        Example: {'target': 'Churn', 'numerical_features': ['Age', 'Tenure'],
                                  'categorical_features': ['Gender', 'Contract']}
    """
    if not EVIDENTLY_INSTALLED:
        logger.warning(f"Evidently AI not installed. Skipping {profile_name} report generation.")
        return

    if reference_data is None or reference_data.empty:
        logger.warning(f"Reference data for {profile_name} report is empty. Skipping report generation.")
        return
    if current_data is None or current_data.empty:
        logger.warning(f"Current data for {profile_name} report is empty. Skipping report generation.")
        return

    logger.info(f"Generating {profile_name} report. Reference shape: {reference_data.shape}, Current shape: {current_data.shape}")

    evidently_column_mapping = None
    if column_mapping:
        try:
            evidently_column_mapping = ColumnMapping()
            evidently_column_mapping.target = column_mapping.get('target')
            evidently_column_mapping.prediction = column_mapping.get('prediction') # Can be single label or array of probabilities
            evidently_column_mapping.datetime = column_mapping.get('datetime')
            evidently_column_mapping.id = column_mapping.get('id')

            # Ensure numerical and categorical features are lists, even if None
            num_features = column_mapping.get('numerical_features', [])
            cat_features = column_mapping.get('categorical_features', [])

            # Filter out any None values from the lists, if any were accidentally included
            evidently_column_mapping.numerical_features = [col for col in num_features if col] if num_features else None
            evidently_column_mapping.categorical_features = [col for col in cat_features if col] if cat_features else None
            
            # Log the actual features being used after potential filtering
            logger.info(f"Using explicit column mapping for drift report: "
                        f"Target='{evidently_column_mapping.target}', "
                        f"Numerical={evidently_column_mapping.numerical_features}, "
                        f"Categorical={evidently_column_mapping.categorical_features}")

        except Exception as e_map:
            logger.warning(f"Error applying explicit column mapping for Evidently: {e_map}. "
                           "Falling back to auto-mapping if possible.", exc_info=True)
            evidently_column_mapping = None # Fallback
    else:
        logger.info("No explicit column mapping provided for drift report, Evidently will attempt auto-mapping.")
        # For auto-mapping to work well, ensure dtypes are correctly set in your DataFrames.

    try:
        # DataDriftPreset provides a comprehensive set of drift metrics
        data_drift_report_suite = Report(metrics=[
            DataDriftPreset(),
            # You can add more specific metrics if needed, e.g., for individual columns:
            # DatasetDriftMetric(),
            # DataDriftTable(),
            # ColumnDriftMetric(column_name='Age'), # Example for a specific column
        ])
        data_drift_report_suite.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=evidently_column_mapping
        )

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        data_drift_report_suite.save_html(report_path)
        logger.info(f"{profile_name} report saved to: {report_path}")

        # Optional: Save as JSON for programmatic access to results
        # report_json_path = report_path.replace(".html", ".json")
        # data_drift_report_suite.save_json(report_json_path)
        # logger.info(f"{profile_name} report (JSON) saved to: {report_json_path}")

    except Exception as e:
        logger.error(f"Failed to generate {profile_name} report: {e}", exc_info=True)

# --- Example usage (can be run as a standalone script for testing) ---
if __name__ == '__main__':
    # This block is for testing the utility functions directly.
    # In a real pipeline, these functions would be called by other scripts (e.g., train_pipeline.py, monitor_pipeline.py).

    # Configure basic logging for standalone script execution
    if not logging.getLogger().hasHandlers() or all(isinstance(h, logging.NullHandler) for h in logging.getLogger().handlers):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("Running utils.py as a standalone script for testing report generation...")

    # Create dummy data for testing
    dummy_current_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 12],
        'feature2': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C', 'A', 'B'],
        'target_variable': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    dummy_reference_data = pd.DataFrame({
        'feature1': [10, 20, 30, 15, 25, 12, 22, 18, 28, 16, 11, 21],
        'feature2': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'A', 'B', 'C', 'A', 'E'], # Introduce drift
        'target_variable': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    # Define paths for test reports
    test_reports_dir = "test_reports" # Create this directory in your project root for test outputs
    os.makedirs(test_reports_dir, exist_ok=True)
    test_quality_report_path = os.path.join(test_reports_dir, "test_data_quality_report.html")
    test_drift_report_path = os.path.join(test_reports_dir, "test_data_drift_report.html")

    # --- Test Data Quality Report ---
    if EVIDENTLY_INSTALLED:
        logger.info("\n--- Testing Data Quality Report Generation ---")
        generate_data_quality_report(
            current_data=dummy_current_data,
            report_path=test_quality_report_path,
            profile_name="Test Data Quality"
        )
    else:
        logger.warning("Skipping Data Quality Report test as Evidently is not installed.")


    # --- Test Data Drift Report ---
    if EVIDENTLY_INSTALLED:
        logger.info("\n--- Testing Data Drift Report Generation ---")
        # Define column mapping for the dummy data
        drift_column_mapping = {
            'target': 'target_variable',
            'numerical_features': ['feature1'],
            'categorical_features': ['feature2']
        }
        generate_data_drift_report(
            reference_data=dummy_reference_data,
            current_data=dummy_current_data,
            report_path=test_drift_report_path,
            profile_name="Test Data Drift",
            column_mapping=drift_column_mapping
        )
    else:
        logger.warning("Skipping Data Drift Report test as Evidently is not installed.")

    logger.info("\nStandalone test script for utils.py finished. Check the 'test_reports' directory.")