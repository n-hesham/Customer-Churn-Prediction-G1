```python
import pytest
from fastapi.testclient import TestClient
from deployment.app import app

client = TestClient(app)

def test_predict_endpoint():
    # Sample input data
    sample_data = {
        "Age": 30,
        "Gender": "Male",
        "Tenure": 12,
        "Usage Frequency": 10,
        "Support Calls": 2,
        "Payment Delay": 5,
        "Subscription Type": "Standard",
        "Contract Length": "Annual",
        "Total Spend": 500,
        "Last Interaction": 10
    }
    
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [0, 1]
```