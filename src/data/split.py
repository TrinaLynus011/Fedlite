import pandas as pd
import numpy as np
import yaml
import os
import json

def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)

def run_split():
    config = load_config()
    raw_path = config["raw_data_path"]
    processed_dir = config["processed_dir"]
    patient_col = config["patient_col"]
    encounter_col = config["encounter_col"]
    target_col = config["target_col"]
    split_config = config["split"]
    seed = split_config["seed"]
    leakage_ids = config["discharge_disposition_leakage_ids"]
    
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Original row count: {len(df)}")
    
    # Target conversion
    # <30 -> 1, >30 -> 0, NO -> 0
    df[target_col] = df[target_col].apply(lambda x: 1 if x == '<30' else 0)
    
    # Filter out deceased and hospice patients to prevent leakage
    print(f"Filtering out discharge disposition IDs: {leakage_ids} (deceased/hospice)...")
    filtered_df = df[~df['discharge_disposition_id'].isin(leakage_ids)].copy()
    print(f"Row count after leakage filtering: {len(filtered_df)}")
    
    # Unique patient splitting
    unique_patients = filtered_df[patient_col].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_patients)
    
    total_patients = len(unique_patients)
    train_idx = int(total_patients * split_config["train_ratio"])
    val_idx = train_idx + int(total_patients * split_config["val_ratio"])
    
    train_patients = set(unique_patients[:train_idx])
    val_patients = set(unique_patients[train_idx:val_idx])
    test_patients = set(unique_patients[val_idx:])
    
    # Verify split integrity
    assert train_patients.isdisjoint(val_patients), "Train and Val patients overlap!"
    assert train_patients.isdisjoint(test_patients), "Train and Test patients overlap!"
    assert val_patients.isdisjoint(test_patients), "Val and Test patients overlap!"
    
    # Separate dataframes
    train_df = filtered_df[filtered_df[patient_col].isin(train_patients)].copy()
    val_df = filtered_df[filtered_df[patient_col].isin(val_patients)].copy()
    test_df = filtered_df[filtered_df[patient_col].isin(test_patients)].copy()
    
    # Verify all filtered rows are accounted for
    assert len(train_df) + len(val_df) + len(test_df) == len(filtered_df), "Mismatch in split row counts!"
    
    print(f"Train rows: {len(train_df)}, unique patients: {len(train_patients)}")
    print(f"Val rows: {len(val_df)}, unique patients: {len(val_patients)}")
    print(f"Test rows: {len(test_df)}, unique patients: {len(test_patients)}")
    
    # Calculate prevalence
    train_prevalence = float((train_df[target_col] == 1).mean())
    val_prevalence = float((val_df[target_col] == 1).mean())
    test_prevalence = float((test_df[target_col] == 1).mean())
    
    # Save splits
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    print("Saved processed splits.")
    
    # Write split_integrity.json
    integrity = {
        "random_seed": seed,
        "splits": {
            "train": {
                "patients": len(train_patients),
                "encounters": len(train_df),
                "positive_prevalence": train_prevalence
            },
            "validation": {
                "patients": len(val_patients),
                "encounters": len(val_df),
                "positive_prevalence": val_prevalence
            },
            "test": {
                "patients": len(test_patients),
                "encounters": len(test_df),
                "positive_prevalence": test_prevalence
            }
        },
        "overlaps": {
            "train_val": len(train_patients.intersection(val_patients)),
            "train_test": len(train_patients.intersection(test_patients)),
            "val_test": len(val_patients.intersection(test_test_overlap := val_patients.intersection(test_patients)))
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/split_integrity.json", "w") as f:
        json.dump(integrity, f, indent=2)
    print("Saved results/split_integrity.json")
    
    # Write docs/methodology/data_leakage.md
    os.makedirs("docs/methodology", exist_ok=True)
    md_content = f"""# Data Leakage & Patient Splitting Methodology

## Background
In the Diabetes 130-US Hospitals dataset, many patients appear multiple times (up to 40 encounters per patient). If a random row-level split is used:
- The same patient's characteristics, clinical history, and baseline risk are split across the training, validation, and test sets.
- The model can memorize patient-specific patterns rather than generalizable diabetes readmission risk factors, creating a severe optimistic bias in performance metrics.
- This represents a critical **methodological failure** that invalidates research.

## Splitting Strategy
To guarantee no patient-level leakage, **FedLite-Care** implements a patient-based splitting protocol:
1. All encounters are mapped to their unique patient identifier (`{patient_col}`).
2. Unique patient IDs are randomly partitioned into Train (70%), Validation (15%), and Test (15%) using a deterministic seed (`{seed}`).
3. Encounters are grouped into the split corresponding to their patient's ID.

## Leakage Prevention Audit

- **Random Seed**: `{seed}`
- **Patient ID Overlap**:
  - Train $\\cap$ Validation: **{integrity['overlaps']['train_val']}** patients
  - Train $\\cap$ Test: **{integrity['overlaps']['train_test']}** patients
  - Validation $\\cap$ Test: **{integrity['overlaps']['val_test']}** patients

| Split | Unique Patients | Encounters | Positive Class Prevalence |
| --- | --- | --- | --- |
| **Train** | {integrity['splits']['train']['patients']:,} | {integrity['splits']['train']['encounters']:,} | {train_prevalence*100:.2f}% |
| **Validation** | {integrity['splits']['validation']['patients']:,} | {integrity['splits']['validation']['encounters']:,} | {val_prevalence*100:.2f}% |
| **Test** | {integrity['splits']['test']['patients']:,} | {integrity['splits']['test']['encounters']:,} | {test_prevalence*100:.2f}% |

## Discharge Disposition Leakage Filtering
Patients who die during hospitalization or are discharged to hospice cannot be readmitted within 30 days (or ever). Including these patients in the negative class introduces statistical leakage.
The following discharge disposition IDs were filtered out prior to splitting:
- `11` (Expired)
- `13` (Hospice / home)
- `14` (Hospice / medical facility)
- `19` (Expired at home)
- `20` (Expired in medical facility)
- `21` (Expired, place unknown)

Total rows removed by this filter: **{len(df) - len(filtered_df):,}** encounters.
"""
    with open("docs/methodology/data_leakage.md", "w") as f:
        f.write(md_content)
    print("Saved docs/methodology/data_leakage.md")

if __name__ == "__main__":
    run_split()
