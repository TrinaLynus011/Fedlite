import xgboost as xgb
import yaml
import time
import os

def load_config():
    with open("configs/models.yaml", "r") as f:
        return yaml.safe_load(f)["xgboost"]

def train_xgboost(X_train, y_train, X_val, y_val, config_override=None):
    config = load_config()
    if config_override:
        config.update(config_override)
        
    print(f"Training Centralized XGBoost (n_estimators={config['n_estimators']}, max_depth={config['max_depth']})...")
    
    # Class weights or scale_pos_weight
    # Typically, scale_pos_weight = sum(negative cases) / sum(positive cases)
    scale_pos_weight = config.get("scale_pos_weight", 1.0)
    
    model = xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        random_state=config["random_state"],
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    training_time = time.time() - start_time
    
    # Validation predictions
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    return model, y_val_prob, training_time
