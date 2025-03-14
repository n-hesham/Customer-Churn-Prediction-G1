# Customer-Churn-Prediction
This repository is for building a machine learning model to predict customer churn. It will include data preprocessing, exploratory analysis, model training, and evaluation. The goal is to identify at-risk customers using predictive analytics.


## 📌 Folder & File Descriptions  

- **`data/`** → Contains raw and processed datasets.  
- **`notebooks/`** → Stores the main Jupyter Notebook (`churn_prediction.ipynb`) where all team members contribute.  
- **`models/`** → Saves trained machine learning models for deployment and further evaluation.  
- **`deployment/`** → Includes files for deploying the model as a **Flask API**.  
- **`results/`** → Stores evaluation metrics, performance visualizations, and reports.  
- **`requirements.txt`** → Contains the necessary Python dependencies for running the project.  
- **`.gitignore`** → Specifies files that should not be tracked by Git (e.g., large datasets, virtual environments).  

This structure ensures **clarity, scalability, and smooth collaboration** throughout the project. 🚀  


## 📂 Customer Churn Prediction - Project Structure  

The repository is organized as follows:
```bash
Customer-Churn-Prediction/  
│── data/                     # Stores datasets  
│   ├── raw/                  # Original dataset  
│   ├── processed/            # Cleaned & preprocessed data  
│  
│── notebooks/                 # Jupyter Notebook for the entire project  
│   ├── churn_prediction.ipynb # Main notebook (everyone works here)  
│  
│── models/                    # Trained machine learning models  
│   ├── final_model.pkl        # Best saved model  
│  
│── deployment/                 # Flask API & deployment files  
│   ├── app.py                 # Flask application for serving predictions  
│  
│── results/                    # Model evaluation & visualizations  
│   ├── confusion_matrix.png    # Confusion matrix visualization  
│   ├── feature_importance.png  # Feature importance graph  
│  
│── requirements.txt            # List of required Python libraries  
│── README.md                   # Project overview & setup instructions  
│── .gitignore                  # Specifies files to ignore in version control  
