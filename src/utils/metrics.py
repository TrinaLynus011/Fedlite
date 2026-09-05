import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no Tk/display required
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

def calculate_metrics(y_true, y_prob, threshold=0.5):
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Risk predictions
    y_pred = (y_prob >= threshold).astype(int)
    
    # Standard metrics
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))
    acc = float((y_pred == y_true).mean())
    
    # Specificity calculation from confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "brier_score": brier,
        "accuracy": acc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

def save_evaluation_plots(y_true, y_prob, experiment_id, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Threshold predictions
    y_pred = (y_prob >= 0.5).astype(int)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_true, y_prob):.4f})", color='#e74c3c', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {experiment_id}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_id}_roc.png"), dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label=f"PR (AUC = {average_precision_score(y_true, y_prob):.4f})", color='#3498db', lw=2)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {experiment_id}")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_id}_pr.png"), dpi=300)
    plt.close()
    
    # 3. Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "s-", label="Calibration", color='#2ecc71', lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Reliability Diagram - {experiment_id}")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_id}_calibration.png"), dpi=300)
    plt.close()
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {experiment_id}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Readmit', 'Readmit'], rotation=45)
    plt.yticks(tick_marks, ['No Readmit', 'Readmit'])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]:,}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight='bold')
                 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, f"{experiment_id}_confusion_matrix.png"), dpi=300)
    plt.close()
