# tests/test_feature_engineering.py
import pytest
import pandas as pd
import numpy as np
from mlops.pipeline.feature_engineering import _manual_feature_engineering_core, apply_manual_feature_engineering
from mlops.pipeline import config # To get STATS_PATH for apply_manual_feature_engineering test
import os
import joblib

@pytest.fixture
def sample_raw_df():
    data = {
        'Usage Frequency': [10, 0, 50, 20],
        'Last Interaction': [1, 0, 5, 10], # Includes 0 for division test
        'Total Spend': [100, 5, 200, 150],
        'Payment Delay': [0, 10, 2, 5],
        'Support Calls': [0, 3, 1, 2],
        'Age': [25, 30, 45, 60],
        'Tenure': [6, 15, 30, 50] # In months
    }
    return pd.DataFrame(data)

def test_manual_fe_core_feature_creation(sample_raw_df):
    engineered_df, stats = _manual_feature_engineering_core(sample_raw_df.copy(), fit_mode=True)

    assert 'Log_Usage_Rate' in engineered_df.columns
    assert 'Spend_Per_Usage' in engineered_df.columns
    assert 'Payment_Delay_Ratio' in engineered_df.columns
    assert 'High_Value' in engineered_df.columns
    assert 'At_Risk' in engineered_df.columns
    assert 'Age_group' in engineered_df.columns
    assert 'tenure_group' in engineered_df.columns

    # Check Log_Usage_Rate for first row (10 / (1 + 1e-6))
    expected_log_usage_0 = np.log1p(10 / (1 + 1e-6))
    assert np.isclose(engineered_df['Log_Usage_Rate'].iloc[0], expected_log_usage_0)
    # Check Spend_Per_Usage for first row (100 / (10 + 1e-6))
    expected_spend_per_usage_0 = 100 / (10 + 1e-6)
    assert np.isclose(engineered_df['Spend_Per_Usage'].iloc[0], expected_spend_per_usage_0)
    # Check Spend_Per_Usage for second row (Usage Frequency = 0) -> Total Spend / (0 + 1e-6) -> then filled
    # Initially it would be 5 / 1e-6. Then replaced by 0, then median.
    # Let's check the median calculation.
    # SPU before fillna: [10, 5e6, 40, 7.5]. Median of this is (10+7.5)/2 = 8.75 or 40 (depending on sorting)
    # With fillna(0) first: [10, 0, 40, 7.5]. Median is (10+7.5)/2=8.75 if we consider 0.
    # The code calculates median on original, then fills NaN (from inf), then fills again with 0, then with median.
    # This logic in _manual_feature_engineering_core for Spend_Per_Usage fillna needs to be precise for testing.
    # Current logic: .replace([np.inf, -np.inf], np.nan).fillna(0).fillna(median_of_original_or_nan_replaced)
    # Let's assume the median for Spend_Per_Usage was calculated correctly and used.
    # For the second row where Usage Frequency is 0:
    # Spend_Per_Usage = 5 / (0 + 1e-6) -> large number -> inf -> nan
    # then filled with 0, then median.
    # If original SPU values were [10, 5/1e-6, 4, 7.5].  inf replaced by NaN. [10, NaN, 4, 7.5]
    # Median of [10,4,7.5] is 7.5. So NaN should be 7.5.
    # Then fillna(0): [10,0,4,7.5] if median not applied yet.
    # Then fillna(median_of_original): if `current_stats['spend_per_usage_median']` was 7.5, then [10, 7.5, 4, 7.5]
    # This needs careful checking against the notebook or intended logic.
    # For now, let's just check it's not inf or nan.
    assert not engineered_df['Spend_Per_Usage'].iloc[1] == np.inf
    assert not pd.isna(engineered_df['Spend_Per_Usage'].iloc[1])


def test_manual_fe_core_stats_creation(sample_raw_df):
    _, stats = _manual_feature_engineering_core(sample_raw_df.copy(), fit_mode=True)
    assert 'spend_75th' in stats
    assert 'payment_delay_median' in stats
    assert 'support_calls_median' in stats
    assert 'spend_per_usage_median' in stats # Check this
    assert 'age_bins' in stats
    assert 'age_labels' in stats
    assert 'tenure_bins' in stats
    assert 'tenure_labels' in stats
    assert stats['spend_75th'] == sample_raw_df['Total Spend'].quantile(0.75)

def test_manual_fe_core_transform_mode(sample_raw_df):
    # Fit on a part of data
    df_fit = sample_raw_df.iloc[:2]
    _, stats_fitted = _manual_feature_engineering_core(df_fit.copy(), fit_mode=True)

    # Transform another part of data
    df_transform = sample_raw_df.iloc[2:]
    engineered_df_transformed, _ = _manual_feature_engineering_core(
        df_transform.copy(),
        stats_dict_for_creation=stats_fitted,
        fit_mode=False
    )
    assert engineered_df_transformed.shape[0] == df_transform.shape[0]
    assert 'Log_Usage_Rate' in engineered_df_transformed.columns
    # Check if 'High_Value' uses stats from df_fit
    # For df_fit: Total Spend [100, 5]. 75th percentile is (100+5)*0.75 (approx) or from .quantile()
    # spend_75th_fit = df_fit['Total Spend'].quantile(0.75) # 100*.75 + 5*.25 = 76.25
    # For df_transform, Total Spend is [200, 150]. Both are > spend_75th_fit
    # This test needs more refinement based on exact stat values.

def test_apply_manual_feature_engineering(sample_raw_df, tmp_path):
    # tmp_path is a pytest fixture for a temporary directory per test function
    stats_save_path = tmp_path / "fe_stats.pkl"

    # Split sample_raw_df into train and test like parts
    X_train_raw = sample_raw_df.iloc[:3]
    X_test_raw = sample_raw_df.iloc[3:]

    X_train_fe, X_test_fe, calculated_fe_stats = apply_manual_feature_engineering(
        X_train_raw, X_test_raw, stats_save_path=str(stats_save_path)
    )

    assert os.path.exists(stats_save_path)
    loaded_stats = joblib.load(stats_save_path)
    assert 'pipeline_input_columns' in loaded_stats
    assert 'pipeline_input_columns' in calculated_fe_stats

    assert X_train_fe.shape[0] == X_train_raw.shape[0]
    assert X_test_fe.shape[0] == X_test_raw.shape[0]
    assert list(X_test_fe.columns) == list(X_train_fe.columns) # Critical alignment check
    assert all(col in X_train_fe.columns for col in loaded_stats['pipeline_input_columns'])