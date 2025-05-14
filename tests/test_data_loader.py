# tests/test_data_loader.py
import pytest
import pandas as pd
import os
from mlops.pipeline import data_loader, config # Import from src package

# Create dummy CSV files for testing
@pytest.fixture(scope="module") # module scope: run once per test module
def create_dummy_csv_files(tmp_path_factory):
    # tmp_path_factory is a pytest fixture for creating temporary directories
    temp_dir = tmp_path_factory.mktemp("data")
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir()

    dummy_train_data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Age': [25, 30, 35, 40, 45],
        'Tenure': [1, 2, 1, 3, 2],
        'Usage Frequency': [10, 15, 12, 20, 18],
        'Support Calls': [1,0,2,1,0],
        'Payment Delay': [5,0,10,0,2],
        'Last Interaction': [10,20,5,15,25],
        'Total Spend': [100,150,120,200,180],
        'Churn': [0, 1, 0, 1, 0]
    }
    dummy_test_data = {
        'CustomerID': [6, 7, 8],
        'Age': [50, 55, 22],
        'Tenure': [4, 5, 1],
        'Usage Frequency': [25, 30, 8],
        'Support Calls': [0,1,0],
        'Payment Delay': [0,5,0],
        'Last Interaction': [30,3,10],
        'Total Spend': [250,300,80],
        'Churn': [0, 1, 0] # Test data also has Churn
    }
    df_train = pd.DataFrame(dummy_train_data)
    df_test = pd.DataFrame(dummy_test_data)

    train_file = raw_dir / "dummy_train.csv"
    test_file = raw_dir / "dummy_test.csv"

    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)

    return str(train_file), str(test_file)

def test_load_and_split_data_successful_load(create_dummy_csv_files):
    train_file, test_file = create_dummy_csv_files
    X_train, X_test, y_train, y_test = data_loader.load_and_split_data(
        train_file_path=train_file,
        test_file_path=test_file,
        test_size=0.25, # Adjusted for small dummy data
        random_state=42
    )
    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty
    assert 'CustomerID' not in X_train.columns
    assert 'CustomerID' not in X_test.columns
    assert 'Churn' not in X_train.columns # Churn should be in y_train
    assert 'Churn' not in X_test.columns  # Churn should be in y_test
    # Test split ratio (approximate due to stratification and small N)
    # Total samples = 5 (train_dummy) + 3 (test_dummy) = 8
    # Test size = 0.25 * 8 = 2. So test should have 2 samples.
    assert len(X_test) == 2
    assert len(X_train) == 6


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        data_loader.load_and_split_data(
            train_file_path="non_existent_train.csv",
            test_file_path="non_existent_test.csv"
        )

def test_load_and_split_data_handles_nan(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("data_nan")
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir()
    
    # Data with NaN
    nan_train_data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Age': [25, None, 35, 40, 45], # NaN here
        'Tenure': [1, 2, 1, 3, 2],
        'Usage Frequency': [10, 15, 12, 20, 18],
        'Support Calls': [1,0,2,1,0],
        'Payment Delay': [5,0,10,0,2],
        'Last Interaction': [10,20,5,15,25],
        'Total Spend': [100,150,120,200,180],
        'Churn': [0, 1, 0, 1, 0]
    }
    df_train_nan = pd.DataFrame(nan_train_data)
    train_file_nan = raw_dir / "nan_train.csv"
    df_train_nan.to_csv(train_file_nan, index=False)
    
    # Use the same file for train and test for simplicity of this test
    X_train, X_test, y_train, y_test = data_loader.load_and_split_data(
        train_file_path=train_file_nan,
        test_file_path=train_file_nan, # Using same file
        test_size=0.25
    )
    # Original data had 5 rows, one with NaN. Concatenated makes 10 rows, 2 with NaN. Dropna leaves 8.
    # Then split 75/25 -> 6 train, 2 test.
    assert len(X_train) + len(X_test) == 8 # (5-1) + (5-1) then split. Total should be 8 non-NaN rows.
                                           # Actually, (5+5)=10 rows, 2 with NaN, so 8 rows remain.
                                           # test_size=0.25 of 8 is 2. So 6 train, 2 test.
    assert X_train['Age'].isnull().sum() == 0
    assert X_test['Age'].isnull().sum() == 0