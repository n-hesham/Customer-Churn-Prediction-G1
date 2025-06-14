import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
from sqlalchemy import text

# إعداد الـ Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/Nour Hesham/Documents/Customer-Churn-Prediction-G1/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

def ensure_directory(directory):
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)
    logging.info(f"Directory ensured: {directory}")

def auto_visualize(df: pd.DataFrame, output_path: str, file_title="Auto_Visualization"):
    """Automatically visualize a DataFrame and save the plot."""
    try:
        sns.set_style("whitegrid")
        df_clean = df.dropna()

        if df_clean.empty:
            logging.warning("No data to visualize.")
            return

        plt.figure(figsize=(12, 6))

        if 'Churn' in df_clean.columns:
            avg_columns = [col for col in df_clean.columns if col.lower().startswith('avg')]
            if avg_columns:
                df_melt = df_clean.melt(id_vars='Churn', value_vars=avg_columns, var_name='Metric', value_name='Value')
                sns.barplot(x='Metric', y='Value', hue='Churn', data=df_melt, palette='Set2')
                plt.title(f'Average Metrics by Churn')
            else:
                sns.countplot(x='Churn', data=df_clean, palette='Set2')
                plt.title('Churn Count')

        elif 'age_group' in df_clean.columns and 'total' in df_clean.columns:
            sns.barplot(x='age_group', y='total', data=df_clean.sort_values('age_group'), palette='Set3')
            plt.title('Age Group Distribution')

        elif 'ContractLength' in df_clean.columns and 'churn_rate' in df_clean.columns:
            sns.barplot(x='ContractLength', y='churn_rate', data=df_clean, palette='Set1')
            plt.title('Churn Rate by Contract Length')

        elif 'Gender' in df_clean.columns and 'churn_rate' in df_clean.columns:
            sns.barplot(x='Gender', y='churn_rate', data=df_clean, palette='Set2')
            plt.title('Churn Rate by Gender')

        else:
            first_col = df_clean.columns[0]
            if pd.api.types.is_numeric_dtype(df_clean[first_col]):
                sns.histplot(df_clean[first_col], bins=30, kde=True, color='skyblue')
                plt.title(f'Histogram of {first_col}')
            else:
                counts = df_clean[first_col].value_counts()
                width = max(10, min(len(counts) * 0.8, 25))
                plt.figure(figsize=(width, 6))
                sns.barplot(x=counts.index, y=counts.values, palette='Set3')
                plt.title(f'Barplot of {first_col}')

        plt.tight_layout()
        ensure_directory(output_path)
        image_file = os.path.join(output_path, f"{file_title}.png")
        plt.savefig(image_file)
        plt.clf()
        plt.close()
        logging.info(f"✅ Visualization saved: {image_file}")

    except Exception as e:
        logging.error(f"❌ Error during visualization: {e}")

def run_sql_and_visualize(sql_folder: str, engine, visuals_folder: str):
    """Execute all SQL files and visualize results."""
    ensure_directory(visuals_folder)
    sql_files = [f for f in os.listdir(sql_folder) if f.endswith('.sql')]

    for file in sql_files:
        file_path = os.path.join(sql_folder, file)
        logging.info(f"📥 Executing {file}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                query = f.read()

            with engine.connect() as connection:
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            if df.empty:
                logging.warning(f"⚠️ No data returned from {file}")
                continue

            logging.info(f"📊 Visualizing results from {file}")
            file_title = file.replace('.sql', '')
            auto_visualize(df, visuals_folder, file_title=file_title)

        except Exception as e:
            logging.error(f"❌ Error executing {file}: {e}")

