import pandas as pd
import numpy as np
import yaml
import os
import json
import matplotlib.pyplot as plt

def load_config():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("configs/federated.yaml", "r") as f:
        fed_config = yaml.safe_load(f)
    config.update(fed_config)
    return config

def partition_data(df, client_count, method="iid", alpha=0.5, seed=42):
    """
    Partitions the dataframe into client_count splits.
    Returns a list of dataframes.
    """
    np.random.seed(seed)
    target_col = "readmitted"
    
    if method == "iid":
        # Shuffle and split evenly using pure pandas to guarantee DataFrames are returned
        shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        chunk_size = int(np.ceil(len(shuffled_df) / client_count))
        partitions = []
        for i in range(client_count):
            start = i * chunk_size
            end = min(start + chunk_size, len(shuffled_df))
            partitions.append(shuffled_df.iloc[start:end].copy())
        return partitions
        
    elif method == "non_iid":
        # Dirichlet partitioning based on labels
        # Group indices by class and copy to ensure they are writable
        class_0_indices = df[df[target_col] == 0].index.values.copy()
        class_1_indices = df[df[target_col] == 1].index.values.copy()
        
        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)
        
        # Draw proportions from Dirichlet distribution
        prop_0 = np.random.dirichlet([alpha] * client_count)
        prop_1 = np.random.dirichlet([alpha] * client_count)
        
        # Determine split boundaries
        split_0 = (np.cumsum(prop_0) * len(class_0_indices)).astype(int)
        split_1 = (np.cumsum(prop_1) * len(class_1_indices)).astype(int)
        
        # Ensure we start at 0
        split_0 = np.insert(split_0, 0, 0)
        split_1 = np.insert(split_1, 0, 0)
        
        # Correct last index to avoid rounding issues
        split_0[-1] = len(class_0_indices)
        split_1[-1] = len(class_1_indices)
        
        client_dfs = []
        for i in range(client_count):
            c0_idx = class_0_indices[split_0[i]:split_0[i+1]]
            c1_idx = class_1_indices[split_1[i]:split_1[i+1]]
            
            # Combine and get subset
            client_idx = np.concatenate([c0_idx, c1_idx])
            np.random.shuffle(client_idx)
            
            client_df = df.loc[client_idx].copy()
            client_dfs.append(client_df)
            
        return client_dfs
    else:
        raise ValueError(f"Unknown partitioning method: {method}")

def generate_client_reports(client_dfs, method, alpha, client_count):
    rows = []
    target_col = "readmitted"
    
    # Store labels distribution for plotting
    plot_data = []
    
    for idx, client_df in enumerate(client_dfs):
        n_samples = len(client_df)
        pos_count = (client_df[target_col] == 1).sum()
        neg_count = n_samples - pos_count
        pos_prevalence = pos_count / n_samples if n_samples > 0 else 0.0
        
        # Collect distributions
        age_dist = client_df['age'].value_counts(normalize=True).to_dict() if 'age' in client_df.columns else {}
        admission_dist = client_df['admission_type_id'].value_counts(normalize=True).to_dict() if 'admission_type_id' in client_df.columns else {}
        
        # Numerical statistics
        hospital_time = float(client_df['time_in_hospital'].mean()) if 'time_in_hospital' in client_df.columns else 0.0
        lab_procs = float(client_df['num_lab_procedures'].mean()) if 'num_lab_procedures' in client_df.columns else 0.0
        
        rows.append({
            "client_id": idx,
            "sample_count": n_samples,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_prevalence": pos_prevalence,
            "time_in_hospital_mean": hospital_time,
            "num_lab_procedures_mean": lab_procs,
            "age_distribution": json.dumps(age_dist),
            "admission_type_distribution": json.dumps(admission_dist)
        })
        
        plot_data.append({
            "client_id": idx,
            "Class 0 (Negative)": neg_count,
            "Class 1 (Positive)": pos_count
        })
        
    stats_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    
    # Save CSV
    filename = f"client_distribution_{method}_c{client_count}"
    if method == "non_iid":
        filename += f"_a{alpha}"
    csv_path = f"results/{filename}.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved client distribution to {csv_path}")
    
    # Save master client distribution CSV (defaults to current)
    stats_df.to_csv("results/client_distribution.csv", index=False)
    
    # Create plot
    plot_df = pd.DataFrame(plot_data).set_index("client_id")
    plt.figure(figsize=(10, 6))
    plot_df.plot(kind='bar', stacked=True, color=['#34495e', '#e74c3c'], ax=plt.gca())
    plt.xlabel("Client ID")
    plt.ylabel("Encounter Count")
    plt.title(f"FedLite-Care Client Distribution - Method: {method.upper()} | Clients: {client_count} | alpha: {alpha}")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plot_path = f"figures/{filename}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Save default client distribution figure
    plt.figure(figsize=(10, 6))
    plot_df.plot(kind='bar', stacked=True, color=['#34495e', '#e74c3c'], ax=plt.gca())
    plt.xlabel("Client ID")
    plt.ylabel("Encounter Count")
    plt.title(f"FedLite-Care Client Distribution ({method.upper()}, alpha={alpha})")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/client_distribution.png", dpi=300)
    plt.close()
    
    print(f"Saved client distribution plot to {plot_path} and figures/client_distribution.png")

if __name__ == "__main__":
    # Test partition script
    config = load_config()
    processed_dir = config["processed_dir"]
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    
    print("Testing IID partitioning with 5 clients...")
    iid_dfs = partition_data(train_df, client_count=5, method="iid")
    generate_client_reports(iid_dfs, "iid", alpha=0.5, client_count=5)
    
    print("Testing Non-IID Dirichlet partitioning (alpha=0.5) with 5 clients...")
    non_iid_dfs = partition_data(train_df, client_count=5, method="non_iid", alpha=0.5)
    generate_client_reports(non_iid_dfs, "non_iid", alpha=0.5, client_count=5)
