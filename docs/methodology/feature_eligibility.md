# Feature Eligibility Policy

The **FedLite-Care** framework maintains a strict feature eligibility policy to ensure clinical correctness, prevent data leakage, and maintain high data quality. Every feature in the raw dataset is classified as one of:
- **ELIGIBLE**: Suitable for training and prediction.
- **POTENTIAL_LEAKAGE**: Feature that contains information about events occurring after the prediction target horizon, or which directly codes target outcomes.
- **EXCLUDED**: Features removed due to extreme missingness, lack of relevance, or model memorization risks.
- **UNKNOWN**: Under review.

## Feature Classifications and Rationales

| Feature Name | Classification | Rationale |
| --- | --- | --- |
| `encounter_id` | **EXCLUDED** | Unique identifier for each hospital encounter. Contains no generalizable clinical signal; including it would lead to model memorization. |
| `patient_nbr` | **EXCLUDED** | Unique patient identifier. High risk of model memorization. Used exclusively for patient-level dataset partitioning and splitting to prevent leakage. |
| `weight` | **EXCLUDED** | Over 96.86% of values are missing. Imputation would introduce extreme noise. |
| `payer_code` | **EXCLUDED** | Non-clinical, administrative billing identifier. Lacks relevance to physiological risk factors of diabetes readmission. High missingness (39.56%). |
| `discharge_disposition_id` | **POTENTIAL_LEAKAGE** / **EXCLUDED** | Represents where the patient went after discharge. Contains codes for death (expired) or hospice transfers (e.g. codes 11, 13, 14, 19, 20, 21). Because these patients cannot be readmitted within 30 days, including them leaks the target. These encounters are filtered out from the dataset, and the column is excluded from model features. |
| `readmitted` | **TARGET** | The target label representing 30-day readmission status. |
| *All other features* | **ELIGIBLE** | Demographic background, admissions characteristics, clinical procedures, diagnoses, and medication prescriptions are considered eligible for predictive modeling. |

## Feature Eligibility Enforcement
The feature eligibility policy is programmatically enforced in `src/preprocessing/pipeline.py` and configured in `configs/preprocessing.yaml`. Any column classified as `EXCLUDED` is dropped prior to model input creation, preventing silent inclusion of leakage-inducing features.
