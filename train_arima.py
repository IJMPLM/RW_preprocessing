import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

def train_arima_demo(data_path, model_save_path):
    print(f"Loading sequential data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # ARIMA is traditionally univariate and fit per time-series sequence.
    # Predicting a binary tabular outcome (like Mortality) using ARIMA is mathematically unsound.
    # In SHAP time-series analyses for ICU data, ARIMA is often used to forecast critical
    # continuous biomarkers (e.g., Heart Rate or MAP) for an individual patient.
    
    # Let's extract the longest sequence patient for this demonstration
    record_counts = df['RecordID'].value_counts()
    patient_id = record_counts.index[0]
    patient_df = df[df['RecordID'] == patient_id].sort_values('Hour_Bin')
    
    # Robust floating point handling
    patient_df = patient_df.replace([np.inf, -np.inf], np.nan)
    for col in patient_df.columns:
        if patient_df[col].dtype == 'float64':
            patient_df[col] = patient_df[col].astype(np.float32)

    # Apply Outlier Clipping (Temp < 30 -> 30)
    for col in patient_df.columns:
        if 'Temp' in col:
            patient_df.loc[patient_df[col] < 30, col] = 30

    # Forecast Heart Rate (HR) using ARIMAX with Temp and MAP as exogenous physiological drivers
    target_series = patient_df['HR'].ffill().bfill()
    exog_series = patient_df[['Temp', 'MAP']].ffill().bfill()
    
    print(f"Training ARIMAX model for patient {patient_id} forecasting HR...")
    
    # Basic ARIMA(1,0,1) model
    model = ARIMA(endog=target_series, exog=exog_series, order=(1, 0, 1))
    fitted_model = model.fit(method_kwargs={'maxiter': 200})
    
    print(f"Model Fit AIC: {fitted_model.aic:.2f}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save the statsmodels object.
    # IMPORTANT: Loading this in R requires `reticulate`. 
    # Statsmodels objects do not have a native cross-language binary format (like safetensors).
    # Exporting as JSON would only provide parts/coefficients, requiring manual rebuild in R.
    # Thus, this is saved as a complete pickled model object to preserve native prediction capabilities.
    joblib.dump(fitted_model, model_save_path)
    print(f"ARIMA model saved to {model_save_path}\n")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'processed_datasets')
    out_dir = os.path.join(base_dir, 'trained_models')
    
    # 1. Mortality (Sequential)
    train_arima_demo(
        os.path.join(data_dir, 'mortality_sequential.csv'),
        os.path.join(out_dir, 'arima_mortality_hr.pkl')
    )
