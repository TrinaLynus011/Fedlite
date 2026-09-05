# Data Leakage & Patient Splitting Methodology

## Background
In the Diabetes 130-US Hospitals dataset, many patients appear multiple times (up to 40 encounters per patient). If a random row-level split is used:
- The same patient's characteristics, clinical history, and baseline risk are split across the training, validation, and test sets.
- The model can memorize patient-specific patterns rather than generalizable diabetes readmission risk factors, creating a severe optimistic bias in performance metrics.
- This represents a critical **methodological failure** that invalidates research.

## Splitting Strategy
To guarantee no patient-level leakage, **FedLite-Care** implements a patient-based splitting protocol:
1. All encounters are mapped to their unique patient identifier (`patient_nbr`).
2. Unique patient IDs are randomly partitioned into Train (70%), Validation (15%), and Test (15%) using a deterministic seed (`42`).
3. Encounters are grouped into the split corresponding to their patient's ID.

## Leakage Prevention Audit

- **Random Seed**: `42`
- **Patient ID Overlap**:
  - Train $\cap$ Validation: **0** patients
  - Train $\cap$ Test: **0** patients
  - Validation $\cap$ Test: **0** patients

| Split | Unique Patients | Encounters | Positive Class Prevalence |
| --- | --- | --- | --- |
| **Train** | 48,993 | 69,665 | 11.52% |
| **Validation** | 10,498 | 14,686 | 10.88% |
| **Test** | 10,499 | 14,992 | 11.28% |

## Discharge Disposition Leakage Filtering
Patients who die during hospitalization or are discharged to hospice cannot be readmitted within 30 days (or ever). Including these patients in the negative class introduces statistical leakage.
The following discharge disposition IDs were filtered out prior to splitting:
- `11` (Expired)
- `13` (Hospice / home)
- `14` (Hospice / medical facility)
- `19` (Expired at home)
- `20` (Expired in medical facility)
- `21` (Expired, place unknown)

Total rows removed by this filter: **2,423** encounters.
