import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import yaml
import time
import os
import sys

def load_config():
    with open("configs/models.yaml", "r") as f:
        return yaml.safe_load(f)["logistic_regression"]

class PyTorchLR(nn.Module):
    """
    Single-layer linear model with sigmoid output, equivalent to Logistic Regression.
    Perfect for federated learning with Flower parameter aggregation.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def train_sklearn_lr(X_train, y_train, X_val, y_val, config_override=None):
    config = load_config()
    if config_override:
        config.update(config_override)
        
    print(f"Training scikit-learn L2 Logistic Regression (C={config['C']})...")
    
    # Class weights for handling imbalance
    class_weight = config.get("class_weight", "balanced")
    
    model = LogisticRegression(
        penalty="l2",
        C=config["C"],
        max_iter=config["max_iter"],
        solver=config["solver"],
        class_weight=class_weight,
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Validation predictions
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    return model, y_val_prob, training_time
