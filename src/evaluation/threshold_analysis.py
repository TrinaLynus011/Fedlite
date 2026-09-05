import os
import pandas as pd
import numpy as np
import torch
import joblib
import yaml
import sys

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

# Add src/ to path
sys.path.append(os.path.abspath("src"))

from models.logistic_regression import train_sklearn_lr
from models.mlp import train_pytorch_mlp
from models.xgboost_baseline import train_xgboost


def evaluate_threshold(y_true, y_prob, threshold):
    """
    Calculate threshold-dependent classification metrics.
    AUROC/AUPRC are intentionally excluded because they
    do not depend on the classification threshold.
    """

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true, y_pred
    )

    accuracy = float(
        (y_pred == y_true).mean()
    )

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )
    else:
        specificity = 0.0

    return {
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
        "Balanced Accuracy": balanced_accuracy,
        "Accuracy": accuracy
    }


def main():

    processed_dir = "data/processed"

    print("Loading preprocessed datasets...")

    train_df = pd.read_csv(
        os.path.join(
            processed_dir,
            "train_preprocessed.csv"
        )
    )

    val_df = pd.read_csv(
        os.path.join(
            processed_dir,
            "val_preprocessed.csv"
        )
    )

    test_df = pd.read_csv(
        os.path.join(
            processed_dir,
            "test_preprocessed.csv"
        )
    )

    target_col = "readmitted"

    features = [
        col for col in train_df.columns
        if col != target_col
    ]

    X_train = train_df[features].values
    y_train = train_df[target_col].values

    X_val = val_df[features].values
    y_val = val_df[target_col].values

    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")

    # --------------------------------------------------
    # Train centralized models
    # --------------------------------------------------

    validation_predictions = {}

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    print("\n=== Logistic Regression ===")

    lr_model, lr_val_prob, _ = train_sklearn_lr(
        X_train,
        y_train,
        X_val,
        y_val
    )

    validation_predictions["CENTRAL_LR"] = lr_val_prob

    # --------------------------------------------------
    # MLP
    # --------------------------------------------------

    print("\n=== MLP ===")

    mlp_model, mlp_val_prob, _ = train_pytorch_mlp(
        X_train,
        y_train,
        X_val,
        y_val
    )

    validation_predictions["CENTRAL_MLP"] = mlp_val_prob

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    print("\n=== XGBoost ===")

    xgb_model, xgb_val_prob, _ = train_xgboost(
        X_train,
        y_train,
        X_val,
        y_val
    )

    validation_predictions["CENTRAL_XGB"] = xgb_val_prob

    # --------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------

    thresholds = np.arange(
        0.05,
        1.00,
        0.05
    )

    results = []

    for model_name, y_prob in validation_predictions.items():

        print(f"\nAnalyzing {model_name}...")

        for threshold in thresholds:

            metrics = evaluate_threshold(
                y_val,
                np.asarray(y_prob),
                float(round(threshold, 2))
            )

            metrics["Model"] = model_name

            results.append(metrics)

    results_df = pd.DataFrame(results)

    # Put Model first
    results_df = results_df[
        [
            "Model",
            "Threshold",
            "Precision",
            "Recall",
            "F1-Score",
            "Specificity",
            "Balanced Accuracy",
            "Accuracy"
        ]
    ]

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    output_path = (
        "results/threshold_analysis.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved threshold analysis to: "
        f"{output_path}"
    )

    # --------------------------------------------------
    # Print best F1 threshold per model
    # --------------------------------------------------

    print("\n=== Best Validation F1 Threshold ===")

    for model_name in validation_predictions:

        model_results = results_df[
            results_df["Model"] == model_name
        ]

        best_row = model_results.loc[
            model_results["F1-Score"].idxmax()
        ]

        print(
            f"\n{model_name}"
        )

        print(
            f"Threshold: "
            f"{best_row['Threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_row['Precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_row['Recall']:.4f}"
        )

        print(
            f"F1: "
            f"{best_row['F1-Score']:.4f}"
        )

        print(
            f"Specificity: "
            f"{best_row['Specificity']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{best_row['Balanced Accuracy']:.4f}"
        )

    # --------------------------------------------------
    # Print threshold = 0.50 for comparison
    # --------------------------------------------------

    print("\n=== Validation Metrics at Threshold 0.50 ===")

    baseline_rows = results_df[
        results_df["Threshold"] == 0.50
    ]

    print(
        baseline_rows.to_string(index=False)
    )


if __name__ == "__main__":
    main()