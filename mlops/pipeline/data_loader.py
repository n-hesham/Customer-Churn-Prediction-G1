# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import logging
from . import config # Assuming config.py is in the same directory or 'src' is the package root

logger = logging.getLogger(__name__)


def load_and_split_data(train_file_path, test_file_path, test_size=0.2, random_state=42):
    """
    Loads data from specified train and test file paths.
    Concatenates them, drops CustomerID, handles basic NaN (drops rows with any NaN),
    and splits into training and testing sets for features (X) and target (y).
    Stratifies the split by the target variable.
    """
    logger.info(f"Data loading process started.")
    logger.info(f"Attempting to load training data from: {train_file_path}")
    logger.info(f"Attempting to load testing data from: {test_file_path}")

    if not os.path.exists(train_file_path):
        logger.error(f"Training data file NOT FOUND at: {train_file_path}")
        raise FileNotFoundError(f"Training data file not found: {train_file_path}")
    if not os.path.exists(test_file_path):
        logger.error(f"Testing data file NOT FOUND at: {test_file_path}")
        raise FileNotFoundError(f"Testing data file not found: {test_file_path}")

    try:
        df_train_raw = pd.read_csv(train_file_path, encoding='ascii', delimiter=',')
        df_test_raw = pd.read_csv(test_file_path, encoding='ascii', delimiter=',')
        logger.info(f"Successfully loaded data. Training data shape (raw): {df_train_raw.shape}, Test data shape (raw): {df_test_raw.shape}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        raise

    df_full = pd.concat([df_train_raw, df_test_raw], axis=0, ignore_index=True)
    logger.info(f"Combined data shape: {df_full.shape}")
    
    initial_rows = len(df_full)
    # Mimic notebook's df = df.dropna()
    df_full = df_full.dropna()
    if len(df_full) < initial_rows:
        dropped_rows = initial_rows - len(df_full)
        logger.warning(f"Dropped {dropped_rows} rows due to any NaN values, matching notebook logic.")
    else:
        logger.info("No rows dropped due to NaN values.")
        
    logger.info(f"Data shape after dropping NaN: {df_full.shape}")

    if 'CustomerID' in df_full.columns:
        df_full = df_full.drop('CustomerID', axis=1)
        logger.info("Dropped 'CustomerID' column.")
    else:
        logger.warning("'CustomerID' column not found, skipping drop.")
    
    if 'Churn' not in df_full.columns:
        logger.error("'Churn' column not found in the dataset after processing. Cannot proceed.")
        raise ValueError("'Churn' column is missing from the combined and preprocessed dataset.")

    X = df_full.drop('Churn', axis=1)
    y = df_full['Churn'] # Assuming 'Churn' is integer 0 or 1
    
    # Stratify by y to ensure similar class proportions in train and test splits
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Data split into train/test: X_train_raw shape {X_train_raw.shape}, X_test_raw shape {X_test_raw.shape}, y_train shape {y_train.shape}, y_test shape {y_test.shape}")
    return X_train_raw, X_test_raw, y_train, y_test