import pandas as pd
from sqlalchemy import create_engine
import logging
import yaml
import os
from src.utils.visualization import run_sql_and_visualize


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

with open('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_data():
    """Load data from MySQL database."""
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://{config['database']['user']}:{config['database']['password']}@{config['database']['host']}/{config['database']['database']}"
        )
        query = config['data']['raw_data_query']
        df = pd.read_sql_query(query, con=engine)
        logging.info("Data loaded successfully from database.")

        run_sql_and_visualize(
         r'C:\Users\Nour Hesham\Documents\Customer-Churn-Prediction-G1\queries',
         engine,
         r'C:\Users\Nour Hesham\Documents\Customer-Churn-Prediction-G1\visuals'
          )
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    df = load_data()
    print(df.head())

