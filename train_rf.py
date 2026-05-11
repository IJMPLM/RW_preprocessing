import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
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
    
    # Apply Outlier Clipping (Temp < 30 -> 30)
    for col in X.columns:
        if 'Temp' in col:
            X.loc[X[col] < 30, col] = 30
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Balanced Random Forest Classifier with Iterative Imputation for {target_col}...")
    # Using IterativeImputer with a tree estimator to prevent overflow from extreme collinearity
    from sklearn.tree import DecisionTreeRegressor
    pipeline = Pipeline([
        ('imputer', IterativeImputer(estimator=DecisionTreeRegressor(max_depth=5, random_state=42), max_iter=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save using joblib. Since this is a scikit-learn pipeline, 
    # it must be loaded in R using the `reticulate` library.
    joblib.dump(pipeline, model_save_path)
    print(f"Random Forest pipeline saved to {model_save_path}\n")

if __name__ == '__main__':
    data_dir = r'd:\Repositories\Thesis\preprocessing\processed_datasets'
    out_dir = r'd:\Repositories\Thesis\preprocessing\trained_models'
    
    # 1. Mortality (Early Prediction)
    train_rf(
        os.path.join(data_dir, 'mortality_early_prediction.csv'),
        'target_In-hospital_death',
        os.path.join(out_dir, 'rf_mortality.pkl')
    )
    
    # 2. Sepsis (Early Prediction)
    train_rf(
        os.path.join(data_dir, 'sepsis_early_prediction.csv'),
        'target_SepsisLabel',
        os.path.join(out_dir, 'rf_sepsis.pkl')
    )
