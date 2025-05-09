```python
import pytest
import os
import joblib
from mlops.pipelines.model_training import train_model

def test_model_training():
    # Run model training
    train_model()
    
    # Check if model file exists
    model_path = "models/trained/best_hgb_model.h5"
    assert os.path.exists(model_path), "Trained model file not found"
    
    # Load and verify model
    model = joblib.load(model_path)
    assert hasattr(model, 'predict'), "Model does not have predict method"
```