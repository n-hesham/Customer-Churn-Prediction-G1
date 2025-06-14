import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import joblib
import logging
import yaml
import os

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

def preprocess_data(df, X_train=None, is_training=True):

    max_val = int(df['Tenure'].max())
    min_val = int(df['Tenure'].min())
    tenure_bins = list(range(min_val, max_val + 12, 12))
    tenure_labels = [f"{i} - {i+11}" for i in tenure_bins[:-1]]
    df['Tenure_group'] = pd.cut(df['Tenure'], bins=tenure_bins, right=False, labels=tenure_labels)

    age_bins = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df['Age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    df.drop(['CustomerID'], axis=1, inplace=True, errors='ignore')

    if is_training:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train['Log_Usage_Rate'] = np.log1p(X_train['UsageFrequency'] / (X_train['LastInteraction'] + 1e-6))
        X_train['Spend_Per_Usage'] = X_train['TotalSpend'] / X_train['UsageFrequency']
        X_train['Payment_Delay_Ratio'] = X_train['PaymentDelay'] / X_train['LastInteraction']

        high_value_threshold = X_train['TotalSpend'].quantile(0.75)
        payment_delay_median = X_train['PaymentDelay'].median()
        support_calls_median = X_train['SupportCalls'].median()

        X_train['High_Value'] = ((X_train['TotalSpend'] > high_value_threshold) &
                                 (X_train['PaymentDelay'] < payment_delay_median)).astype(int)
        X_train['At_Risk'] = ((X_train['SupportCalls'] > support_calls_median) &
                              (X_train['PaymentDelay'] > payment_delay_median)).astype(int)

        stats = {
            'high_value_threshold': high_value_threshold,
            'payment_delay_median': payment_delay_median,
            'support_calls_median': support_calls_median
        }
        joblib.dump(stats, config['preprocessing']['stats_path'])

        X_test['Log_Usage_Rate'] = np.log1p(X_test['UsageFrequency'] / (X_test['LastInteraction'] + 1e-6))
        X_test['Spend_Per_Usage'] = X_test['TotalSpend'] / X_test['UsageFrequency']
        X_test['Payment_Delay_Ratio'] = X_test['PaymentDelay'] / X_test['LastInteraction']
        X_test['High_Value'] = ((X_test['TotalSpend'] > high_value_threshold) &
                                (X_test['PaymentDelay'] < payment_delay_median)).astype(int)
        X_test['At_Risk'] = ((X_test['SupportCalls'] > support_calls_median) &
                             (X_test['PaymentDelay'] > payment_delay_median)).astype(int)

        X_train['Spend_Per_Usage'] = X_train['Spend_Per_Usage'].replace([np.inf, -np.inf], np.nan).fillna(X_train['Spend_Per_Usage'].median())
        X_test['Spend_Per_Usage'] = X_test['Spend_Per_Usage'].replace([np.inf, -np.inf], np.nan).fillna(X_train['Spend_Per_Usage'].median())

        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_cols = config['preprocessing']['categorical_columns']
        X_train_encoded = ohe.fit_transform(X_train[cat_cols])
        X_test_encoded = ohe.transform(X_test[cat_cols])

        cat_columns = ohe.get_feature_names_out(cat_cols)
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=cat_columns, index=X_train.index)
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=cat_columns, index=X_test.index)

        joblib.dump(ohe, config['preprocessing']['encoder_path'])

        X_train_final = pd.concat([X_train.drop(cat_cols, axis=1), X_train_encoded], axis=1)
        X_test_final = pd.concat([X_test.drop(cat_cols, axis=1), X_test_encoded], axis=1)

        scaler = StandardScaler()
        num_cols = config['preprocessing']['numerical_columns']
        X_train_final[num_cols] = scaler.fit_transform(X_train_final[num_cols])
        X_test_final[num_cols] = scaler.transform(X_test_final[num_cols])

        joblib.dump(scaler, config['preprocessing']['scaler_path'])

        X_train_final = X_train_final.dropna()
        y_train = y_train.loc[X_train_final.index]
        undersampler = RandomUnderSampler(random_state=42)
        X_train_final, y_train = undersampler.fit_resample(X_train_final, y_train)

        X_with_target = X_train_final.copy()
        X_with_target['Churn'] = y_train


        pd.concat([X_train_final.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(
            config['data']['train_processed'], index=False)
        pd.concat([X_test_final.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
            config['data']['test_processed'], index=False)

        logging.info("Preprocessing completed and data saved.")
        return X_train_final, X_test_final, y_train, y_test, X_train

    else:
        stats = joblib.load(config['preprocessing']['stats_path'])
        ohe = joblib.load(config['preprocessing']['encoder_path'])
        scaler = joblib.load(config['preprocessing']['scaler_path'])

        df['Log_Usage_Rate'] = np.log1p(df['UsageFrequency'] / (df['LastInteraction'] + 1e-6))
        df['Spend_Per_Usage'] = np.where(df['UsageFrequency'] > 0, df['TotalSpend'] / df['UsageFrequency'], 0)
        df['Payment_Delay_Ratio'] = np.where(df['LastInteraction'] > 0, df['PaymentDelay'] / df['LastInteraction'], 0)
        df['High_Value'] = ((df['TotalSpend'] > stats['high_value_threshold']) &
                            (df['PaymentDelay'] < stats['payment_delay_median'])).astype(int)
        df['At_Risk'] = ((df['SupportCalls'] > stats['support_calls_median']) &
                         (df['PaymentDelay'] > stats['payment_delay_median'])).astype(int)

        cat_cols = config['preprocessing']['categorical_columns']
        X_new_encoded = ohe.transform(df[cat_cols])
        cat_columns = ohe.get_feature_names_out(cat_cols)
        X_new_encoded = pd.DataFrame(X_new_encoded, columns=cat_columns, index=df.index)

        X_new_final = pd.concat([df.drop(cat_cols, axis=1), X_new_encoded], axis=1)
        num_cols = config['preprocessing']['numerical_columns']
        X_new_final[num_cols] = scaler.transform(X_new_final[num_cols])

        logging.info("Inference preprocessing completed.")
        return X_new_final

if __name__ == "__main__":
    from src.pipelines.data_ingestion import load_data
    df = load_data()
    X_train_final, X_test_final, y_train, y_test, X_train = preprocess_data(df)
    print(X_train_final.head())