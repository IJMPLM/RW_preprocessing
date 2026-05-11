import pandas as pd
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
    fitted_model = model.fit()
    
    print(f"Model Fit AIC: {fitted_model.aic:.2f}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save the statsmodels object.
    # IMPORTANT: Loading this in R requires `reticulate`. 
    # However, if your SHAP variants are heavily reliant on R objects, it is HIGHLY recommended 
    # to fit ARIMA models directly in R using `forecast::auto.arima` instead of passing
    # Python statsmodels objects across the language barrier.
    joblib.dump(fitted_model, model_save_path)
    print(f"ARIMA model saved to {model_save_path}\n")

if __name__ == '__main__':
    data_dir = r'd:\Repositories\Thesis\preprocessing\processed_datasets'
    out_dir = r'd:\Repositories\Thesis\preprocessing\trained_models'
    
    # 1. Mortality (Sequential)
    train_arima_demo(
        os.path.join(data_dir, 'mortality_sequential.csv'),
        os.path.join(out_dir, 'arima_mortality_hr.pkl')
    )
