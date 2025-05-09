```python
import pytest
import pandas as pd
from mlops.pipelines.data_preprocessing import preprocess_data

def test_data_preprocessing():
    # Run preprocessing
    preprocess_data()
    
    # Check if processed files exist
    train_path = "data/processed/train_processed.csv"
    test_path = "data/processed/test_processed.csv"
    
    assert os.path.exists(train_path), "Processed train file not found"
    assert os.path.exists(test_path), "Processed test file not found"
    
    # Load and verify data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    assert 'Churn' in train_df.columns, "Churn column missing in train data"
    assert 'Churn' in test_df.columns, "Churn column missing in test data"
    assert train_df.isnull().sum().sum() == 0, "Missing values in train data"
    assert test_df.isnull().sum().sum() == 0, "Missing values in test data"
```