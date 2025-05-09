```python
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import yaml
import logging
import os
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

# Setup logging
def setup_logging():
    config = load_config()
    logging.basicConfig(
        filename=config['monitoring']['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config():
    with open("config/project_config.yaml", "r") as file:
        return yaml.safe_load(file)

# Load preprocessing objects and model
config = load_config()
ohe = joblib.load(os.path.join(config['models']['preprocessing'], 'One_Hot_Encoder.pkl'))
scaler = joblib.load(os.path.join(config['models']['preprocessing'], 'standard_scaler.pkl'))
stats = joblib.load(os.path.join(config['models']['preprocessing'], 'feature_engineering_stats.pkl'))
model = joblib.load(os.path.join(config['models']['path'], 'best_hgb_model.h5'))

# Define input schema
class CustomerData(BaseModel):
    Age: int
    Gender: str
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: int

def preprocess_input(data: Dict):
    df = pd.DataFrame([data])
    
    # Create features
    df['Log_Usage_Rate'] = np.log1p(df['Usage_Frequency'] / (df.Last_Interaction + 1e-6))
    df['Spend_Per_Usage'] = df['Total_Spend'] / df.Usage_Frequency.replace(0, np.nan)
    df['Payment_Delay_Ratio'] = df['Payment_Delay'] / df.Last_Interaction.replace(0, np.nan)
    
    df['High_Value'] = ((df['Total_Spend'] > stats['spend_75th']) &
                        (df['Payment_Delay'] < stats['payment_delay_median'])).astype(int)
    df['At_Risk'] = ((df['Support_Calls'] > stats['support_calls_median']) &
                     (df['Payment_Delay'] > stats['payment_delay_median'])).astype(int)
    
    df['Age_group'] = pd.cut(df['Age'], bins=stats['age_bins'], labels=['Young', 'Adult', 'Senior', 'Elder'])
    df['tenure_group'] = pd.cut(df['Tenure'], bins=stats['tenure_bins'], labels=['<1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr'])
    
    df['Spend_Per_Usage'] = df['Spend_Per_Usage'].replace([np.inf, -np.inf], np.nan).fillna(df['Spend_Per_Usage'].median())
    df['Payment_Delay_Ratio'] = df['Payment_Delay_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(df['Payment_Delay_Ratio'].median())
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    encoded_data = pd.DataFrame(
        ohe.transform(df[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=df.index
    )
    
    # Combine and scale
    final_data = pd.concat([df.drop(cat_cols, axis=1), encoded_data], axis=1)
    final_data[num_cols] = scaler.transform(final_data[num_cols])
    
    return final_data

@app.post("/predict")
async def predict(data: CustomerData):
    setup_logging()
    try:
        # Convert input to dictionary and preprocess
        input_data = data.dict()
        processed_data = preprocess_input(input_data)
        
        # Predict
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        logging.info(f"Prediction made: {prediction}, Probability: {probability}")
        return {"prediction": int(prediction), "probability": float(probability)}
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```