# src/feature_engineering.py
import pandas as pd
import numpy as np
import joblib
import logging
from . import config # Use relative import

logger = logging.getLogger(__name__)

def _manual_feature_engineering_core(df, stats_dict_for_creation=None, fit_mode=True):
    """Core logic for manual feature engineering, aligned with the notebook."""
    engineered_df = df.copy()

    # Ensure essential columns for calculations exist, fill with 0 if not (e.g., for partial API calls)
    # These are the columns used in notebook's feature creation
    expected_raw_cols = ['Usage Frequency', 'Last Interaction', 'Total Spend',
                         'Payment Delay', 'Support Calls', 'Age', 'Tenure']
    for col in expected_raw_cols:
        if col not in engineered_df.columns:
            logger.warning(f"Column '{col}' not found in input for manual FE. Adding with 0.")
            engineered_df[col] = 0

    # Clip before calculations to avoid negative inputs where not logical
    engineered_df['Usage Frequency'] = engineered_df['Usage Frequency'].clip(lower=0)
    engineered_df['Last Interaction'] = engineered_df['Last Interaction'].clip(lower=0) # Ensure non-negative
    engineered_df['Total Spend'] = engineered_df['Total Spend'].clip(lower=0)


    # Feature creation as in the notebook
    engineered_df['Log_Usage_Rate'] = np.log1p(engineered_df['Usage Frequency'] / (engineered_df['Last Interaction'] + 1e-6))
    # Add 1e-6 to Usage Frequency as well in Spend_Per_Usage to avoid division by zero if Usage Frequency is 0
    engineered_df['Spend_Per_Usage'] = engineered_df['Total Spend'] / (engineered_df['Usage Frequency'] + 1e-6)
    engineered_df['Payment_Delay_Ratio'] = engineered_df['Payment Delay'] / (engineered_df['Last Interaction'] + 1e-6)

    # Handle inf/-inf that might result from division by near-zero
    engineered_df['Spend_Per_Usage'] = engineered_df['Spend_Per_Usage'].replace([np.inf, -np.inf], np.nan)
    engineered_df['Payment_Delay_Ratio'] = engineered_df['Payment_Delay_Ratio'].replace([np.inf, -np.inf], np.nan)


    current_stats = {}
    if fit_mode:
        logger.info("Calculating stats for manual feature engineering (fit_mode=True).")
        current_stats['spend_75th'] = engineered_df['Total Spend'].quantile(0.75)
        current_stats['payment_delay_median'] = engineered_df['Payment Delay'].median()
        current_stats['support_calls_median'] = engineered_df['Support Calls'].median()
        
        # For Spend_Per_Usage, fill NaN with 0 first, then calculate median, as in notebook
        # Note: notebook used fillna(0) then fillna(median) - second fillna on median of already 0-filled.
        # We'll calculate median on potentially NaN-containing series then use it for fillna.
        # If we want to exactly match notebook's X_train_demo['Spend_Per_Usage'].median() after initial fillna(0):
        temp_spu_for_median_calc = engineered_df['Spend_Per_Usage'].copy()
        # If the notebook first fills with 0 and then takes median:
        # current_stats['spend_per_usage_median'] = temp_spu_for_median_calc.fillna(0).median()
        # If the notebook fills with 0, then fills NaN resulting from inf with median of the original data:
        current_stats['spend_per_usage_median'] = engineered_df['Spend_Per_Usage'].median() # Median of original calculation before filling infs with 0


        # Binning as in the notebook's feature engineering section
        current_stats['age_bins'] = [0, 25, 45, 65, 100]
        current_stats['age_labels'] = ['Young', 'Adult', 'Senior', 'Elder']
        current_stats['tenure_bins'] = [0, 12, 24, 36, 48, 60] # Matched to notebook
        current_stats['tenure_labels'] = ['<1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr'] # Matched to notebook
    else:
        logger.info("Applying manual feature engineering using provided stats (fit_mode=False).")
        if stats_dict_for_creation is None:
            logger.error("stats_dict_for_creation must be provided when fit_mode is False.")
            raise ValueError("stats_dict_for_creation is required for transform mode.")
        current_stats = stats_dict_for_creation

    engineered_df['High_Value'] = ((engineered_df['Total Spend'] > current_stats['spend_75th']) &
                                   (engineered_df['Payment Delay'] < current_stats['payment_delay_median'])).astype(int)
    engineered_df['At_Risk'] = ((engineered_df['Support Calls'] > current_stats['support_calls_median']) &
                                (engineered_df['Payment Delay'] > current_stats['payment_delay_median'])).astype(int)

    # Ensure Age and Tenure are numeric before pd.cut
    engineered_df['Age'] = pd.to_numeric(engineered_df['Age'], errors='coerce')
    engineered_df['Tenure'] = pd.to_numeric(engineered_df['Tenure'], errors='coerce')

    # Apply binning
    engineered_df['Age_group'] = pd.cut(engineered_df['Age'], bins=current_stats['age_bins'],
                                        labels=current_stats['age_labels'], right=False, include_lowest=True)
    engineered_df['tenure_group'] = pd.cut(engineered_df['Tenure'], bins=current_stats['tenure_bins'],
                                           labels=current_stats['tenure_labels'], right=False, include_lowest=True)

    # Handle NaN in Spend_Per_Usage and Payment_Delay_Ratio after inf replacement
    # Notebook logic: .replace([np.inf, -np.inf], np.nan).fillna(0)
    # Then later: .replace([np.inf, -np.inf], np.nan).fillna(X_train['Spend_Per_Usage'].median())
    # This implies filling NaNs (including those from inf) with 0, then for Spend_Per_Usage, potentially overriding
    # those 0s with a median if the second fillna was intended for NaNs *not* yet filled.
    # For simplicity and robustness, we'll fill NaNs (from inf or original) with 0, then median for Spend_Per_Usage.
    
    # First fill with 0 (as notebook did first for Spend_Per_Usage)
    engineered_df['Spend_Per_Usage'] = engineered_df['Spend_Per_Usage'].fillna(0)
    # Then fill with median (if it was calculated, for Spend_Per_Usage)
    if 'spend_per_usage_median' in current_stats:
         engineered_df['Spend_Per_Usage'] = engineered_df['Spend_Per_Usage'].fillna(current_stats['spend_per_usage_median'])
    else: # if in transform mode and somehow median wasn't saved, default to 0
        engineered_df['Spend_Per_Usage'] = engineered_df['Spend_Per_Usage'].fillna(0)


    engineered_df['Payment_Delay_Ratio'] = engineered_df['Payment_Delay_Ratio'].fillna(0) # Notebook implies fillna(0) for this

    # Convert categorical groups to string and fill NaNs (e.g., if Age/Tenure was NaN)
    engineered_df['Age_group'] = engineered_df['Age_group'].astype(str).fillna('Unknown')
    engineered_df['tenure_group'] = engineered_df['tenure_group'].astype(str).fillna('Unknown')

    return engineered_df, current_stats


def apply_manual_feature_engineering(X_train_raw, X_test_raw, stats_save_path=config.STATS_PATH):
    """
    Applies manual feature engineering to training data (fitting stats) and
    test data (using fitted stats). Saves the calculated stats.
    Aligns columns of the test set to match the training set after FE.
    """
    logger.info("Starting manual feature engineering process...")

    # Apply to training data (fit mode)
    X_train_fe, calculated_fe_stats = _manual_feature_engineering_core(X_train_raw.copy(), fit_mode=True)

    # Store the columns *after* manual FE; these are input to the scikit-learn pipeline
    calculated_fe_stats['pipeline_input_columns'] = X_train_fe.columns.tolist()
    logger.info(f"Manual FE on X_train complete. Shape: {X_train_fe.shape}")
    logger.debug(f"Columns for scikit-learn pipeline input: {calculated_fe_stats['pipeline_input_columns']}")

    # Save feature engineering stats
    joblib.dump(calculated_fe_stats, stats_save_path)
    logger.info(f"Feature engineering stats saved to {stats_save_path}")

    # Apply to test data (transform mode)
    X_test_fe, _ = _manual_feature_engineering_core(
        X_test_raw.copy(),
        stats_dict_for_creation=calculated_fe_stats,
        fit_mode=False
    )
    logger.info(f"Manual FE on X_test complete. Shape: {X_test_fe.shape}")

    # Align columns of X_test_fe to match X_train_fe exactly, based on 'pipeline_input_columns'
    expected_cols = calculated_fe_stats['pipeline_input_columns']
    
    # Add any missing columns to X_test_fe
    for col in expected_cols:
        if col not in X_test_fe.columns:
            logger.warning(f"Aligning X_test_fe: Column '{col}' missing from test set after FE. Adding with default.")
            # Determine default value based on X_train_fe's dtype for that column
            train_col_dtype = X_train_fe[col].dtype
            if pd.api.types.is_string_dtype(train_col_dtype) or \
               pd.api.types.is_object_dtype(train_col_dtype) or \
               pd.api.types.is_categorical_dtype(train_col_dtype):
                X_test_fe[col] = "Unknown" # Default for string/object/category
            elif pd.api.types.is_numeric_dtype(train_col_dtype):
                X_test_fe[col] = 0.0 # Default for numeric
            else:
                X_test_fe[col] = pd.NA # Fallback, though less likely with prior handling
    
    # Reorder and select columns to match training set exactly
    X_test_fe = X_test_fe[expected_cols]
    logger.info(f"X_test_fe columns aligned and reordered. Final shape: {X_test_fe.shape}")

    return X_train_fe, X_test_fe, calculated_fe_stats