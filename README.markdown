# Customer Churn Prediction Pipeline

## Table of Contents
- [Customer Churn Prediction Pipeline](#customer-churn-prediction-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Making Batch Predictions](#making-batch-predictions)
    - [Monitoring Model and Data](#monitoring-model-and-data)
    - [Triggering Retraining](#triggering-retraining)
  - [Configuration](#configuration)
  - [Notes](#notes)
  - [Contributing](#contributing)
  - [Contact](#contact)

## Overview
This project provides a machine learning pipeline for predicting customer churn based on customer attributes. It follows MLOps best practices, integrating data processing, feature engineering, model training, batch inference, performance monitoring, and automated retraining. The pipeline uses a scikit-learn `HistGradientBoostingClassifier`, MLflow for experiment tracking and model management, and Evidently AI for data quality and drift analysis.

## Features
- **Data Loading**: Loads and preprocesses raw data, handling missing values and dropping irrelevant columns (e.g., `CustomerID`).
- **Feature Engineering**: Creates domain-specific features like `Log_Usage_Rate`, `Spend_Per_Usage`, and `High_Value`.
- **Model Training**: Builds a scikit-learn pipeline with preprocessing (OneHotEncoder, StandardScaler) and a classifier, logging results to MLflow.
- **Batch Inference**: Generates predictions on new data using models from MLflow or local storage.
- **Monitoring**: Tracks model performance (AUC, accuracy) and data drift, generating Evidently reports.
- **Retraining**: Triggers retraining based on performance thresholds or manual invocation, with model promotion logic.

## Prerequisites
- Python 3.8 or higher
- Required Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `mlflow`
  - `joblib`
  - `matplotlib`
  - `seaborn`
  - `evidently` (optional, for data reports)
- Dataset files: `customer_churn_dataset-training-master.csv` and `customer_churn_dataset-testing-master.csv`
- MLflow tracking server (optional, defaults to local `mlruns` directory)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` with the above packages and run:
   ```bash
   pip install pandas numpy scikit-learn mlflow joblib matplotlib seaborn evidently
   ```

4. **Configure MLflow (Optional)**:
   Set the `MLFLOW_TRACKING_URI` for a remote server:
   ```bash
   export MLFLOW_TRACKING_URI=http://<mlflow-server>:5000
   ```
   Or update `MLFLOW_TRACKING_URI` in `mlops/pipeline/config.py`.

5. **Prepare Dataset**:
   Place `customer_churn_dataset-training-master.csv` and `customer_churn_dataset-testing-master.csv` in `data/raw/`.

## Project Structure
```
project_root/
├───data
│   ├───predictions
│   ├───processed
│   │       test_processed.csv
│   │       train_processed.csv
│   │
│   ├───raw
│   │       customer_churn_dataset-testing-master.csv
│   │       customer_churn_dataset-training-master.csv
│   │
│   └───reports
├───mlops
│   │   monitor_pipeline.py
│   │   retrain_trigger.py
│   │
│   ├───pipeline
│   │   │   config.py
│   │   │   data_loader.py
│   │   │   feature_engineering.py
│   │   │   model_trainer.py
│   │   │   predict_script.py
│   │   │   train_pipeline.py
│   │   │   utils.py
│   │   │   __init__.py
│
├───mlruns
│   ├───.trash
│   ├───0
│   │       meta.yaml
│   │
│   ├───741110547360237600
│   │   │   meta.yaml
│   │   │
│   │   ├───33ec83b827244aeb82015ff6b923641f
│   │   │   │   meta.yaml
│   │   │   │
│   │   │   ├───artifacts
│   │   │   │   │   classification_report_test.json
│   │   │   │   │   confusion_matrix_test.png
│   │   │   │   │
│   │   │   │   └───hgb-churn-full-sklearn-pipeline
│   │   │   │           conda.yaml
│   │   │   │           input_example.json
│   │   │   │           MLmodel
│   │   │   │           model.pkl
│   │   │   │           python_env.yaml
│   │   │   │           requirements.txt
│   │   │   │           serving_input_example.json
│   │   │   │
│   │   │   ├───metrics
│   │   │   │       test_accuracy
│   │   │   │       test_auc_score
│   │   │   │
│   │   │   ├───params
│   │   │   │       classifier_categorical_features
│   │   │   │       classifier_class_weight
│   │   │   │       classifier_early_stopping
│   │   │   │       classifier_interaction_cst
│   │   │   │       classifier_l2_regularization
│   │   │   │       classifier_learning_rate
│   │   │   │       classifier_loss
│   │   │   │       classifier_max_bins
│   │   │   │       classifier_max_depth
│   │   │   │       classifier_max_features
│   │   │   │       classifier_max_iter
│   │   │   │       classifier_max_leaf_nodes
│   │   │   │       classifier_min_samples_leaf
│   │   │   │       classifier_monotonic_cst
│   │   │   │       classifier_n_iter_no_change
│   │   │   │       classifier_random_state
│   │   │   │       classifier_scoring
│   │   │   │       classifier_tol
│   │   │   │       classifier_validation_fraction
│   │   │   │       classifier_verbose
│   │   │   │       classifier_warm_start
│   │   │   │       fe_pipeline_input_cols_count
│   │   │   │       model_type
│   │   │   │       sklearn_cat_features_ohe_count
│   │   │   │       sklearn_num_features_scaled_count
│   │   │   │
│   │   │   └───tags
│   │   │           mlflow.log-model.history
│   │   │           mlflow.runName
│   │   │           mlflow.source.git.commit
│   │   │           mlflow.source.name
│   │   │           mlflow.source.type
│   │   │           mlflow.user
│   │   │
│   │   └───ac0656e9ea0b442687c947decf625a9c
│   │       │   meta.yaml
│   │       │
│   │       ├───artifacts
│   │       │   │   classification_report_test.json
│   │       │   │   confusion_matrix_test.png
│   │       │   │
│   │       │   └───hgb-churn-full-sklearn-pipeline

│   │       │
│   │       ├───metrics
│   │       │       test_accuracy
│   │       │       test_auc_score
│   │       │
│   │       ├───params

│   │       │
│   │       └───tags

│   │
│   └───models
│       └───ChurnPredictionHGB_Refactored
│           │   meta.yaml
│           │
│           ├───version-1
│           │       meta.yaml
│           │
│           └───version-2
│                   meta.yaml
│
├───models
│   │   best_hgb_model_pipeline.pkl
│   │
│   └───saved_preprocessing
│           feature_engineering_stats.pkl
│
├───notebooks
│       churn_prediction.ipynb
│
├───results
│       1.png
│       10.png
│       11.png
│
├───tests
│   │   test_data_loader.py
│   │   test_feature_engineering.py
│   │
│   └───__pycache__
│           test_feature_engineering.cpython-313-pytest-8.3.5.pyc
│
└───__pycache__
        pup.cpython-313.pyc
└── README.md                   # Project documentation
```

## Usage

### Training the Model
Run the training pipeline:
```bash
python -m mlops.pipeline.train_pipeline
```
**Outputs**:
- Processed data: `data/processed/train_processed.csv`, `data/processed/test_processed.csv`
- Model: `models/best_hgb_model_pipeline.pkl`
- Feature stats: `models/saved_preprocessing/feature_engineering_stats.pkl`
- MLflow logs: Metrics (AUC, accuracy), artifacts (confusion matrix)
- Reports (if enabled): `data/reports/initial_train_raw_data_quality_report.html`, `data/reports/processed_train_test_drift_report.html`

### Making Batch Predictions
Generate predictions:
```bash
python -m mlops.pipeline.predict_script \
  --input_data data/raw/customer_churn_dataset-testing-master.csv \
  --output_preds data/predictions/test_predictions.csv \
  --fe_stats_path models/saved_preprocessing/feature_engineering_stats.pkl \
  --model_name ChurnPredictionHGB_Refactored \
  --model_stage Production \
  --local_model_path models/best_lgb_model.pkl
```
**Output**: Predictions in `data/predictions/test_predictions.csv` with `CustomerID` (if present), `Predicted_Churn_Label`, and `Predicted_Churn_Probability`.

### Monitoring Model and Data
Monitor performance and drift:
```bash
python -m mlops.monitor_pipeline
```
**Outputs**:
- MLflow metrics: `live_auc`, `live_accuracy`, `live_f1_churn` in `Churn_Prediction_Refactored_Pipeline_Monitoring` experiment
- Drift report: `data/reports/live_data_drift_report.html` (if Evidently installed)
- Alerts: Logged if AUC < 0.70 or >3 features drift

### Triggering Retraining
Check and trigger retraining:
```bash
python -m mlops.retrain_trigger
```
**Options**:
- `--force`: Force retraining
**Logic**:
- Retrains if production AUC < 0.65 or no production model exists
- Promotes new model if AUC improves by >0.01

## Configuration
Edit `mlops/pipeline/config.py` for:
- **Paths**: `DATA_DIR`, `MODELS_DIR`, `TRAIN_DATA_PATH`, etc.
- **MLflow**: `MLFLOW_EXPERIMENT_NAME` (`Churn_Prediction_Refactored_Pipeline`), `MLFLOW_MODEL_REGISTRY_NAME` (`ChurnPredictionHGB_Refactored`)
- **Reports**: `GENERATE_DATA_REPORTS` (True/False)
- **Logging**: Console output (FileHandler commented out)

Set `MLFLOW_TRACKING_URI` via environment variable or `config.py` for remote tracking.

## Notes
- **Evidently**: Requires `evidently` for reports; skipped with warnings if not installed.
- **Monitoring Placeholders**: `monitor_pipeline.py` uses mock data for production samples and ground truth. Replace with actual data sources in production.
- **MLflow**: Local tracking uses `mlruns/`. Ensure server access for remote tracking.
- **Dataset**: Ensure raw data files are in `data/raw/` with expected columns (e.g., `Churn`, `Age`, `Tenure`).

## Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature/<feature-name>`
3. Commit changes: `git commit -m "Add <feature>"`
4. Push: `git push origin feature/<feature-name>`
5. Open a pull request.

Follow PEP 8 and include logging in contributions.

## Contact
For issues or questions, open a GitHub issue or contact the maintainers.