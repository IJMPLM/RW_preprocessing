import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

# Standard Target Variables
TARGET_VARIABLES = [
    'Age', 'Sex', 'AdmissionType',
    'HR', 'RR', 'Temp', 'SBP', 'MAP', 'SpO2', 'SaO2',
    'Creatinine', 'BUN', 'Sodium', 'Potassium', 'Bicarbonate', 'Lactate', 'pH', 'PaO2', 'FiO2', 'WBC', 'Platelets', 'Hemoglobin', 'Bilirubin', 'AST', 'ALT', 'Glucose',
    'GCS'
]

# --- 1. Mortality Dataset (PhysioNet 2012) Logic ---
MORTALITY_PARAM_MAPPING = {
    'Age': 'Age', 'Gender': 'Sex', 'ICUType': 'AdmissionType',
    'HR': 'HR', 'RespRate': 'RR', 'Temp': 'Temp', 
    'NISysABP': 'SBP', 'SysABP': 'SBP', 'NIMAP': 'MAP', 'MAP': 'MAP', 
    'SpO2': 'SpO2', 'SaO2': 'SaO2',
    'Creatinine': 'Creatinine', 'BUN': 'BUN', 'Na': 'Sodium', 'K': 'Potassium',
    'HCO3': 'Bicarbonate', 'Lactate': 'Lactate', 'pH': 'pH', 'PaO2': 'PaO2', 'FiO2': 'FiO2',
    'WBC': 'WBC', 'Platelets': 'Platelets', 'HCT': 'Hemoglobin', 'Bilirubin': 'Bilirubin',
    'AST': 'AST', 'ALT': 'ALT', 'Glucose': 'Glucose', 'GCS': 'GCS'
}

def parse_mortality_txt(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        if not {'Time', 'Parameter', 'Value'}.issubset(set(df.columns)):
            return pd.DataFrame()
            
        record_id = os.path.basename(filepath).replace('.txt', '')
        record_row = df[df['Parameter'] == 'RecordID']
        if not record_row.empty:
            record_id = record_row['Value'].iloc[0]
            
        df['RecordID'] = int(record_id) if str(record_id).isdigit() else record_id
        df['Parameter'] = df['Parameter'].map(MORTALITY_PARAM_MAPPING)
        df = df.dropna(subset=['Parameter'])
        df = df[df['Parameter'].isin(TARGET_VARIABLES)]
        
        def parse_time(time_str):
            if pd.isna(time_str): return np.nan
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return int(parts[0]) + int(parts[1]) / 60.0
            return float(time_str)
            
        df['Hour'] = df['Time'].apply(parse_time)
        return df
    except Exception:
        return pd.DataFrame()

def process_mortality_file(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = parse_mortality_txt(filepath)
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    record_id = df['RecordID'].iloc[0]
    
    # Sequential
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Clean Outliers
    df.loc[(df['Parameter'] == 'Temp') & (df['Value'] < 25), 'Value'] = np.nan
    df.loc[(df['Parameter'] == 'HR') & (df['Value'] <= 0.0), 'Value'] = np.nan
    df.loc[(df['Parameter'] == 'Sex') & (df['Value'] < 0), 'Value'] = np.nan
    df = df.dropna(subset=['Value'])
    
    df['Hour_Bin'] = np.floor(df['Hour']).astype(int)
    seq_grouped = df.groupby(['RecordID', 'Hour_Bin', 'Parameter'])['Value'].mean().reset_index()
    seq_df = seq_grouped.pivot(index=['RecordID', 'Hour_Bin'], columns='Parameter', values='Value')
    
    if not seq_df.empty:
        max_hour = seq_df.index.get_level_values('Hour_Bin').max()
        full_index = pd.MultiIndex.from_product([[record_id], range(0, max_hour + 1)], names=['RecordID', 'Hour_Bin'])
        seq_df = seq_df.reindex(full_index).ffill()
        
    def create_tabular(df_win):
        stats = []
        if not df_win.empty:
            for param, group in df_win.groupby('Parameter'):
                values = pd.to_numeric(group['Value'], errors='coerce').dropna()
                if not values.empty:
                    stats.append({
                        'RecordID': record_id, 'Parameter': param,
                        'Min': values.min(), 'Max': values.max(), 'Mean': values.mean(),
                        'Variance': values.var() if len(values) > 1 else 0.0, 'Last': values.iloc[-1]
                    })
        if not stats: return pd.DataFrame()
        tdf = pd.DataFrame(stats).pivot(index='RecordID', columns='Parameter', values=['Min', 'Max', 'Mean', 'Variance', 'Last'])
        tdf.columns = [f"{col[1]}_{col[0]}" for col in tdf.columns]
        return tdf.reset_index()

    # Tabular (Full)
    tab_df_full = create_tabular(df)
    
    # Tabular (Early Prediction 24h)
    df_window_24 = df[df['Hour'] <= 24].copy()
    tab_df_24 = create_tabular(df_window_24)
        
    return tab_df_full, tab_df_24, seq_df

# --- 2. Sepsis Dataset (PhysioNet 2019) Logic ---
SEPSIS_PARAM_MAPPING = {
    'HR': 'HR', 'Resp': 'RR', 'Temp': 'Temp', 'SBP': 'SBP', 'MAP': 'MAP', 
    'O2Sat': 'SpO2', 'SaO2': 'SaO2',
    'Creatinine': 'Creatinine', 'BUN': 'BUN', 'Potassium': 'Potassium', 'Chloride': 'Sodium', # Approx
    'HCO3': 'Bicarbonate', 'Lactate': 'Lactate', 'pH': 'pH', 'FiO2': 'FiO2',
    'WBC': 'WBC', 'Platelets': 'Platelets', 'Hgb': 'Hemoglobin', 'Bilirubin_total': 'Bilirubin',
    'AST': 'AST', 'Glucose': 'Glucose', 'Age': 'Age', 'Gender': 'Sex'
}

def process_sepsis_file(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        df = pd.read_csv(filepath, sep='|')
        record_id = os.path.basename(filepath).replace('.psv', '')
        
        # Sepsis files are ALREADY sequential
        seq_df = df.rename(columns=SEPSIS_PARAM_MAPPING)
        seq_df['RecordID'] = record_id
        seq_df['Hour_Bin'] = seq_df['ICULOS'].astype(int) - 1 # ICULOS starts at 1
        
        # Outcome is SepsisLabel
        outcome_col = seq_df[['RecordID', 'Hour_Bin', 'SepsisLabel']].copy()
        
        seq_cols = [c for c in seq_df.columns if c in TARGET_VARIABLES]
        seq_final = seq_df[['RecordID', 'Hour_Bin'] + seq_cols].set_index(['RecordID', 'Hour_Bin']).ffill()
        # Add outcome back to sequential
        seq_final['target_SepsisLabel'] = outcome_col.set_index(['RecordID', 'Hour_Bin'])['SepsisLabel']
        
        def create_sepsis_tabular(df_win):
            tab_data = {'RecordID': [record_id]}
            for col in seq_cols:
                vals = df_win[col].dropna()
                if not vals.empty:
                    tab_data[f"{col}_Min"] = [vals.min()]
                    tab_data[f"{col}_Max"] = [vals.max()]
                    tab_data[f"{col}_Mean"] = [vals.mean()]
                    tab_data[f"{col}_Variance"] = [vals.var() if len(vals) > 1 else 0.0]
                    tab_data[f"{col}_Last"] = [vals.iloc[-1]]
            # Get outcome for tabular (Did they ever have sepsis?)
            tab_data['target_SepsisLabel'] = [df['SepsisLabel'].max()]
            return pd.DataFrame(tab_data)

        # Tabular (Full)
        tab_df_full = create_sepsis_tabular(seq_df)
        
        # Tabular (first 24h)
        df_window = seq_df[seq_df['Hour_Bin'] <= 24].copy()
        tab_df_24 = create_sepsis_tabular(df_window)
        
        return tab_df_full, tab_df_24, seq_final
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- Orchestration ---
def main():
    base_dir = r'd:\Repositories\Thesis\datasets'
    out_dir = r'd:\Repositories\Thesis\preprocessing\processed_datasets'
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Process Mortality Dataset
    mortality_dir = os.path.join(base_dir, 'Predicting Mortality of ICU Patients')
    if os.path.exists(mortality_dir):
        print(f"Processing Mortality Dataset...")
        files = glob.glob(os.path.join(mortality_dir, '**', '*.txt'), recursive=True)
        # Exclude Outcomes files from parsing
        files = [f for f in files if 'Outcomes' not in os.path.basename(f)]
        # De-duplicate file paths just in case
        files = list(set(files))
        
        tab_full_dfs, tab_early_dfs, seq_dfs = [], [], []
        with ProcessPoolExecutor() as executor:
            for t_f, t_e, s in executor.map(process_mortality_file, files):
                if not t_f.empty: tab_full_dfs.append(t_f)
                if not t_e.empty: tab_early_dfs.append(t_e)
                if not s.empty: seq_dfs.append(s)
                
        if tab_full_dfs:
            tab_full_df = pd.concat(tab_full_dfs, ignore_index=True).drop_duplicates(subset=['RecordID'])
            tab_early_df = pd.concat(tab_early_dfs, ignore_index=True).drop_duplicates(subset=['RecordID'])
            seq_df = pd.concat(seq_dfs)
            seq_df = seq_df[~seq_df.index.duplicated(keep='first')]
            
            # Load Outcomes
            outcomes_files = glob.glob(os.path.join(mortality_dir, 'Outcomes-*.txt'))
            if outcomes_files:
                out_df = pd.concat([pd.read_csv(f) for f in outcomes_files])
                out_df = out_df.drop_duplicates(subset=['RecordID'])
                
                # Fix Placeholders
                out_df.loc[out_df['Survival'] == -1, 'Survival'] = 3650
                out_df.loc[out_df['Survival'] < 0, 'Survival'] = np.nan
                
                # Merge outcomes, but prefix with target_ to prevent leakage
                out_df = out_df.rename(columns={'In-hospital_death': 'target_In-hospital_death', 'Survival': 'target_Survival'})
                
                tab_full_df = pd.merge(tab_full_df, out_df[['RecordID', 'target_In-hospital_death', 'target_Survival']], on='RecordID', how='left')
                tab_early_df = pd.merge(tab_early_df, out_df[['RecordID', 'target_In-hospital_death', 'target_Survival']], on='RecordID', how='left')
                
                seq_df = seq_df.reset_index()
                seq_df = pd.merge(seq_df, out_df[['RecordID', 'target_In-hospital_death', 'target_Survival']], on='RecordID', how='left')
                seq_df = seq_df.set_index(['RecordID', 'Hour_Bin'])
            
            # Drop zero-variance columns (Optimization for Kernel SHAP)
            for col in list(tab_full_df.columns):
                if col != 'RecordID' and tab_full_df[col].nunique() <= 1:
                    tab_full_df.drop(columns=[col], inplace=True)
            for col in list(tab_early_df.columns):
                if col != 'RecordID' and tab_early_df[col].nunique() <= 1:
                    tab_early_df.drop(columns=[col], inplace=True)
            for col in list(seq_df.columns):
                if col not in ['RecordID', 'Hour_Bin'] and seq_df[col].nunique() <= 1:
                    seq_df.drop(columns=[col], inplace=True)

            tab_full_df.to_csv(os.path.join(out_dir, 'mortality_tabular.csv'), index=False)
            tab_early_df.to_csv(os.path.join(out_dir, 'mortality_early_prediction.csv'), index=False)
            seq_df.fillna(seq_df.mean()).to_csv(os.path.join(out_dir, 'mortality_sequential.csv'))
            print(f"Mortality dataset saved.")

    # 2. Process Sepsis Dataset
    sepsis_dir = os.path.join(base_dir, 'Early Prediction of Sepsis from Clinical Data')
    if os.path.exists(sepsis_dir):
        print(f"Processing Sepsis Dataset...")
        files = glob.glob(os.path.join(sepsis_dir, '**', '*.psv'), recursive=True)
        files = list(set(files))
        
        tab_full_dfs, tab_early_dfs, seq_dfs = [], [], []
        with ProcessPoolExecutor() as executor:
            for t_f, t_e, s in executor.map(process_sepsis_file, files):
                if not t_f.empty: tab_full_dfs.append(t_f)
                if not t_e.empty: tab_early_dfs.append(t_e)
                if not s.empty: seq_dfs.append(s)
                
        if tab_full_dfs:
            tab_full_df = pd.concat(tab_full_dfs, ignore_index=True).drop_duplicates(subset=['RecordID'])
            tab_early_df = pd.concat(tab_early_dfs, ignore_index=True).drop_duplicates(subset=['RecordID'])
            seq_df = pd.concat(seq_dfs)
            seq_df = seq_df[~seq_df.index.duplicated(keep='first')]
            
            # Drop zero-variance columns (Optimization for Kernel SHAP)
            for col in list(tab_full_df.columns):
                if col != 'RecordID' and tab_full_df[col].nunique() <= 1:
                    tab_full_df.drop(columns=[col], inplace=True)
            for col in list(tab_early_df.columns):
                if col != 'RecordID' and tab_early_df[col].nunique() <= 1:
                    tab_early_df.drop(columns=[col], inplace=True)
            for col in list(seq_df.columns):
                if col not in ['RecordID', 'Hour_Bin'] and seq_df[col].nunique() <= 1:
                    seq_df.drop(columns=[col], inplace=True)
            
            tab_full_df.to_csv(os.path.join(out_dir, 'sepsis_tabular.csv'), index=False)
            tab_early_df.to_csv(os.path.join(out_dir, 'sepsis_early_prediction.csv'), index=False)
            seq_df.fillna(seq_df.mean()).to_csv(os.path.join(out_dir, 'sepsis_sequential.csv'))
            print(f"Sepsis dataset saved.")

    # 3. MIMIC-I
    print("Note: MIMIC-I dataset was skipped as it contains unstructured alarm logs rather than structured tables.")

if __name__ == '__main__':
    main()
