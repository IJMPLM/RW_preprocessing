import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

def train_rf(data_path, target_col, model_save_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    # Separate features and target
    target_cols = [c for c in df.columns if c.startswith('target_')]
    X = df.drop(columns=['RecordID'] + target_cols, errors='ignore')
    y = df[target_col]
    
    # Robust floating point handling
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.astype(np.float32)
    
    # Apply Outlier Clipping (Temp < 30 -> 30)
    for col in X.columns:
        if 'Temp' in col:
            X.loc[X[col] < 30, col] = 30
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate scale_pos_weight for class imbalance
    pos_count = y_train.sum()
    scale_pos_weight = (len(y_train) - pos_count) / pos_count if pos_count > 0 else 1.0
    
    print(f"Training Balanced XGBoost Random Forest Classifier for {target_col}...")
    
    try:
        model = xgb.XGBRFClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            tree_method='hist',
            device='cuda'
        )
        model.fit(X_train, y_train)
        print("Successfully trained using GPU.")
    except Exception as e:
        print(f"GPU training failed: {e}\nFalling back to CPU...")
        model = xgb.XGBRFClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            tree_method='hist',
            device='cpu'
        )
        model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save as JSON. This is the universally supported format for XGBoost models across
    # Python, R, and other bindings without generating UBJSON fallback warnings.
    model.save_model(model_save_path)
    print(f"Random Forest model saved to {model_save_path}\n")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'processed_datasets')
    out_dir = os.path.join(base_dir, 'trained_models')
    
    # 1. Mortality (Early Prediction)
    train_rf(
        os.path.join(data_dir, 'mortality_early_prediction.csv'),
        'target_In-hospital_death',
        os.path.join(out_dir, 'rf_mortality.json')
    )
    
    # 2. Sepsis (Early Prediction)
    train_rf(
        os.path.join(data_dir, 'sepsis_early_prediction.csv'),
        'target_SepsisLabel',
        os.path.join(out_dir, 'rf_sepsis.json')
    )
