# Dataset Audit Report: FedLite-Care

This report contains a comprehensive programmatic audit of the raw dataset used for the **FedLite-Care** diabetes readmission prediction task.

## Key Dataset Metrics

| Metric | Value |
| --- | --- |
| **Dataset Name** | Diabetes 130-US Hospitals for Years 1999–2008 |
| **Filename** | `diabetes_130_raw.csv` |
| **File Size** | 18.27 MB (19159383 bytes) |
| **Number of Rows (Encounters)** | 101,766 |
| **Number of Columns (Features)** | 50 |
| **Unique Patients** | 71,518 |
| **Unique Encounters** | 101,766 |
| **Duplicate Rows** | 0 |
| **Duplicate Encounters** | 0 |
| **Duplicate Patient IDs** | 30248 |
| **Patients with Multiple Encounters** | 16773 (23.45%) |
| **Overall Cell Missingness** | 7.35% (374,017 / 5,088,300 cells) |

## Target Distribution (`readmitted`)

The target variable in the raw dataset is `readmitted`, which contains three distinct values representing different readmission horizons:

| Class | Count | Percentage |
| --- | --- | --- |
| `NO` | 54,864 | 53.91% |
| `>30` | 35,545 | 34.93% |
| `<30` | 11,357 | 11.16% |

For the binary prediction task:
- **Positive Class (`1`)**: `<30` (readmitted within 30 days) -> **11,357** encounters (11.16%)
- **Negative Class (`0`)**: `>30` or `NO` (not readmitted or readmitted after 30 days) -> **90,409** encounters (88.84%)

This shows a **substantial class imbalance**, which must be addressed using appropriate evaluation metrics (AUROC, AUPRC, F1, Sensitivity, Specificity) rather than raw accuracy.

## Missingness Analysis

The following features contain missing data (either represented as standard nulls/NaNs or as `'?'` in the raw CSV):

| Column | Null Count | '?' Count | Total Missing | Percentage Missing |
| --- | --- | --- | --- | --- |
| `weight` | 0 | 98,569 | 98,569 | 96.86% |
| `max_glu_serum` | 96,420 | 0 | 96,420 | 94.75% |
| `A1Cresult` | 84,748 | 0 | 84,748 | 83.28% |
| `medical_specialty` | 0 | 49,949 | 49,949 | 49.08% |
| `payer_code` | 0 | 40,256 | 40,256 | 39.56% |
| `race` | 0 | 2,273 | 2,273 | 2.23% |
| `diag_3` | 0 | 1,423 | 1,423 | 1.40% |
| `diag_2` | 0 | 358 | 358 | 0.35% |
| `diag_1` | 0 | 21 | 21 | 0.02% |

### Critical Findings:
- **`weight`** is missing in **96.86%** of encounters. It must be excluded from feature modeling due to high missingness.
- **`medical_specialty`** is missing in **49.08%** of encounters. It has high missingness and high cardinality.
- **`payer_code`** is missing in **39.56%** of encounters. This feature is administrative and is a candidate for exclusion.
- Diagnosis codes (`diag_1`, `diag_2`, `diag_3`) have minor missingness, but because they represent ICD-9 codes, missing values must be handled carefully.

## Numerical Ranges

The dataset contains the following numerical features:

| Feature | Min | Max | Mean | Std | Median |
| --- | --- | --- | --- | --- | --- |
| `admission_type_id` | 1.0 | 8.0 | 2.02 | 1.45 | 1.0 |
| `discharge_disposition_id` | 1.0 | 28.0 | 3.72 | 5.28 | 1.0 |
| `admission_source_id` | 1.0 | 25.0 | 5.75 | 4.06 | 7.0 |
| `time_in_hospital` | 1.0 | 14.0 | 4.40 | 2.99 | 4.0 |
| `num_lab_procedures` | 1.0 | 132.0 | 43.10 | 19.67 | 44.0 |
| `num_procedures` | 0.0 | 6.0 | 1.34 | 1.71 | 1.0 |
| `num_medications` | 1.0 | 81.0 | 16.02 | 8.13 | 15.0 |
| `number_outpatient` | 0.0 | 42.0 | 0.37 | 1.27 | 0.0 |
| `number_emergency` | 0.0 | 76.0 | 0.20 | 0.93 | 0.0 |
| `number_inpatient` | 0.0 | 21.0 | 0.64 | 1.26 | 0.0 |
| `number_diagnoses` | 1.0 | 16.0 | 7.42 | 1.93 | 8.0 |

## Categorical Cardinality

The dataset contains the following categorical features:

| Feature | Cardinality (Raw) | Cardinality (No '?') | Sample Values |
| --- | --- | --- | --- |
| `race` | 6 | 5 | `Caucasian`, `AfricanAmerican`, `?`, `Other`, `Asian` |
| `gender` | 3 | 3 | `Female`, `Male`, `Unknown/Invalid` |
| `age` | 10 | 10 | `[0-10)`, `[10-20)`, `[20-30)`, `[30-40)`, `[40-50)` |
| `weight` | 10 | 9 | `?`, `[75-100)`, `[50-75)`, `[0-25)`, `[100-125)` |
| `payer_code` | 18 | 17 | `?`, `MC`, `MD`, `HM`, `UN` |
| `medical_specialty` | 73 | 72 | `Pediatrics-Endocrinology`, `?`, `InternalMedicine`, `Family/GeneralPractice`, `Cardiology` |
| `diag_1` | 717 | 716 | `250.83`, `276`, `648`, `8`, `197` |
| `diag_2` | 749 | 748 | `?`, `250.01`, `250`, `250.43`, `157` |
| `diag_3` | 790 | 789 | `?`, `255`, `V27`, `403`, `250` |
| `max_glu_serum` | 4 | 3 | `nan`, `>300`, `Norm`, `>200` |
| `A1Cresult` | 4 | 3 | `nan`, `>7`, `>8`, `Norm` |
| `metformin` | 4 | 4 | `No`, `Steady`, `Up`, `Down` |
| `repaglinide` | 4 | 4 | `No`, `Up`, `Steady`, `Down` |
| `nateglinide` | 4 | 4 | `No`, `Steady`, `Down`, `Up` |
| `chlorpropamide` | 4 | 4 | `No`, `Steady`, `Down`, `Up` |
| `glimepiride` | 4 | 4 | `No`, `Steady`, `Down`, `Up` |
| `acetohexamide` | 2 | 2 | `No`, `Steady` |
| `glipizide` | 4 | 4 | `No`, `Steady`, `Up`, `Down` |
| `glyburide` | 4 | 4 | `No`, `Steady`, `Up`, `Down` |
| `tolbutamide` | 2 | 2 | `No`, `Steady` |
| `pioglitazone` | 4 | 4 | `No`, `Steady`, `Up`, `Down` |
| `rosiglitazone` | 4 | 4 | `No`, `Steady`, `Up`, `Down` |
| `acarbose` | 4 | 4 | `No`, `Steady`, `Up`, `Down` |
| `miglitol` | 4 | 4 | `No`, `Steady`, `Down`, `Up` |
| `troglitazone` | 2 | 2 | `No`, `Steady` |
| `tolazamide` | 3 | 3 | `No`, `Steady`, `Up` |
| `examide` | 1 | 1 | `No` |
| `citoglipton` | 1 | 1 | `No` |
| `insulin` | 4 | 4 | `No`, `Up`, `Steady`, `Down` |
| `glyburide-metformin` | 4 | 4 | `No`, `Steady`, `Down`, `Up` |
| `glipizide-metformin` | 2 | 2 | `No`, `Steady` |
| `glimepiride-pioglitazone` | 2 | 2 | `No`, `Steady` |
| `metformin-rosiglitazone` | 2 | 2 | `No`, `Steady` |
| `metformin-pioglitazone` | 2 | 2 | `No`, `Steady` |
| `change` | 2 | 2 | `No`, `Ch` |
| `diabetesMed` | 2 | 2 | `No`, `Yes` |
| `readmitted` | 3 | 3 | `NO`, `>30`, `<30` |

## Potential Methodological Issues & Data Leakage Risks

1. **Patient Multi-Encounter Bias**: Over 20% of patients have multiple encounters (up to 40 encounters for a single patient). A random split will cause severe patient leakage (the same patient's data appearing in both train and test sets). Splitting **must** be performed at the patient-level (`patient_nbr`).
2. **Discharge Status Leakage**: Patients discharged with certain status codes (e.g., expired/deceased, hospice transfer) cannot be readmitted. Including these encounters in the dataset leaks the readmission target (since readmission is impossible). The preprocessing pipeline must handle these codes.
