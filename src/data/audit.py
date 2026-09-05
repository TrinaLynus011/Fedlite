import pandas as pd
import numpy as np
import json
import os

def audit_dataset():
    csv_path = r"c:\Users\joant\OneDrive\Desktop\My FIles\Semesters\Honors\Mini Project_Honors\Project\Dataset\diabetes_130_raw.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    print("Loading dataset for audit...")
    df = pd.read_csv(csv_path)
    num_rows, num_cols = df.shape
    file_size = os.path.getsize(csv_path)
    
    encounter_col = 'encounter_id'
    patient_col = 'patient_nbr'
    target_col = 'readmitted'
    
    unique_encounters = df[encounter_col].nunique()
    unique_patients = df[patient_col].nunique()
    
    duplicate_rows = int(df.duplicated().sum())
    duplicate_encounters = int(df.duplicated(subset=[encounter_col]).sum())
    duplicate_patients_count = int(df.duplicated(subset=[patient_col]).sum())
    
    patient_encounter_counts = df[patient_col].value_counts()
    patients_with_multiple_encounters = int((patient_encounter_counts > 1).sum())
    
    missing_by_col = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        q_count = (df[col] == '?').sum() if df[col].dtype == object else 0
        total_missing = int(null_count + q_count)
        missing_by_col[col] = {
            "null_count": int(null_count),
            "question_mark_count": int(q_count),
            "total_missing": total_missing,
            "missing_percentage": float(total_missing / num_rows * 100)
        }
        
    target_dist = df[target_col].value_counts().to_dict()
    target_dist = {str(k): int(v) for k, v in target_dist.items()}
    dtypes_dict = {col: str(df[col].dtype) for col in df.columns}
    
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col in [encounter_col, patient_col]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    categorical_cardinality = {}
    for col in categorical_cols:
        unique_vals = df[col].unique()
        unique_vals_clean = [v for v in unique_vals if v != '?' and not pd.isnull(v)]
        categorical_cardinality[col] = {
            "cardinality_raw": int(len(unique_vals)),
            "cardinality_clean": int(len(unique_vals_clean)),
            "sample_values": [str(v) for v in unique_vals[:5]]
        }
        
    numerical_ranges = {}
    for col in numeric_cols:
        desc = df[col].describe()
        numerical_ranges[col] = {
            "min": float(desc['min']),
            "max": float(desc['max']),
            "mean": float(desc['mean']),
            "std": float(desc['std']),
            "median": float(df[col].median())
        }
        
    discharge_dispositions = df['discharge_disposition_id'].value_counts().to_dict()
    discharge_dispositions = {int(k): int(v) for k, v in discharge_dispositions.items()}
    
    audit_summary = {
        "dataset_name": "Diabetes 130-US Hospitals for Years 1999-2008",
        "file_name": os.path.basename(csv_path),
        "file_size_bytes": file_size,
        "rows": num_rows,
        "columns": num_cols,
        "unique_patients": unique_patients,
        "unique_encounters": unique_encounters,
        "duplicate_rows": duplicate_rows,
        "duplicate_encounters": duplicate_encounters,
        "duplicate_patients": duplicate_patients_count,
        "patients_with_multiple_encounters": patients_with_multiple_encounters,
        "target_distribution": target_dist,
        "data_types": dtypes_dict,
        "missingness": missing_by_col,
        "categorical_cardinality": categorical_cardinality,
        "numerical_ranges": numerical_ranges,
        "discharge_disposition_distribution": discharge_dispositions
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/dataset_audit.json", "w") as f:
        json.dump(audit_summary, f, indent=2)
    print("Saved results/dataset_audit.json")
    
    # Save the target distribution separately as required in Section 4
    target_dist_path = os.path.join("results", "target_distribution.json")
    binary_dist = {
        "1": int(target_dist.get('<30', 0)),
        "0": int(target_dist.get('>30', 0) + target_dist.get('NO', 0))
    }
    with open(target_dist_path, "w") as f:
        json.dump({
            "raw": target_dist,
            "binary_30day_readmitted": binary_dist
        }, f, indent=2)
    print("Saved results/target_distribution.json")

if __name__ == "__main__":
    audit_dataset()
