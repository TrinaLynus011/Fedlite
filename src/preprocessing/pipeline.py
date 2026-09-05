import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder

def load_config():
    with open("configs/preprocessing.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("configs/data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    config.update(data_config)
    return config

def fit_transform_pipeline():
    config = load_config()
    processed_dir = config["processed_dir"]
    target_col = config["target_col"]
    
    print("Loading split datasets...")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    num_features = config["numerical_features"]
    cat_features = config["categorical_features"]
    
    # Cast categorical features to string to prevent numerical IDs from being treated as numeric
    for df in [train_df, val_df, test_df]:
        # Replace '?' with None to represent missingness
        df.replace('?', np.nan, inplace=True)
        for col in cat_features:
            df[col] = df[col].astype(str).replace('nan', np.nan)
            
    # X and y split
    X_train = train_df[num_features + cat_features]
    y_train = train_df[target_col]
    
    X_val = val_df[num_features + cat_features]
    y_val = val_df[target_col]
    
    X_test = test_df[num_features + cat_features]
    y_test = test_df[target_col]
    
    # Define preprocessing steps
    # Numerical: median imputation, standard scaling
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical: impute with a constant "missing" class, target encoding, standard scaling
    seed = config["split"]["seed"]
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # target encoder calculates the smoothed target mean for categories
        ('encoder', TargetEncoder(random_state=seed)),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ],
        remainder='drop'
    )
    
    print("Fitting preprocessor pipeline on training data...")
    # Fit on training set only to prevent leakage
    preprocessor.fit(X_train, y_train)
    
    # Save the preprocessor
    os.makedirs("artifacts/preprocessors", exist_ok=True)
    preprocessor_path = "artifacts/preprocessors/preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved fitted preprocessor to {preprocessor_path}")
    
    # Transform datasets
    print("Transforming train, validation, and test datasets...")
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    # Get all feature names
    all_features = num_features + cat_features
    
    # Create transformed dataframes
    train_trans_df = pd.DataFrame(X_train_trans, columns=all_features)
    train_trans_df[target_col] = y_train.values
    
    val_trans_df = pd.DataFrame(X_val_trans, columns=all_features)
    val_trans_df[target_col] = y_val.values
    
    test_trans_df = pd.DataFrame(X_test_trans, columns=all_features)
    test_trans_df[target_col] = y_test.values
    
    # Save processed datasets
    train_trans_df.to_csv(os.path.join(processed_dir, "train_preprocessed.csv"), index=False)
    val_trans_df.to_csv(os.path.join(processed_dir, "val_preprocessed.csv"), index=False)
    test_trans_df.to_csv(os.path.join(processed_dir, "test_preprocessed.csv"), index=False)
    print("Saved preprocessed split datasets.")
    
    # Let's save the input feature dimension to configs/models.yaml dynamically
    models_config_path = "configs/models.yaml"
    with open(models_config_path, "r") as f:
        models_config = yaml.safe_load(f)
    
    models_config["mlp"]["input_dim"] = int(X_train_trans.shape[1])
    with open(models_config_path, "w") as f:
        yaml.safe_dump(models_config, f)
    print(f"Updated models config input_dim to {X_train_trans.shape[1]}")

if __name__ == "__main__":
    fit_transform_pipeline()
