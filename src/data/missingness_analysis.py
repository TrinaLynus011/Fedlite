import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_missingness():
    csv_path = r"c:\Users\joant\OneDrive\Desktop\My FIles\Semesters\Honors\Mini Project_Honors\Project\Dataset\diabetes_130_raw.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    print("Loading dataset for missingness analysis...")
    df = pd.read_csv(csv_path)
    num_rows = len(df)
    
    missing_data = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        q_count = (df[col] == '?').sum() if df[col].dtype == object else 0
        total_missing = int(null_count + q_count)
        pct_missing = float(total_missing / num_rows * 100)
        
        missing_data.append({
            "Feature": col,
            "Null_Count": null_count,
            "Question_Mark_Count": q_count,
            "Total_Missing": total_missing,
            "Percentage_Missing": pct_missing
        })
        
    missing_df = pd.DataFrame(missing_data)
    missing_df = missing_df.sort_values(by="Percentage_Missing", ascending=False)
    
    # Save CSV
    os.makedirs("results", exist_ok=True)
    missing_csv_path = "results/missingness.csv"
    missing_df.to_csv(missing_csv_path, index=False)
    print(f"Saved missingness table to {missing_csv_path}")
    
    # Create plot
    # Only plot columns with missingness > 0 to keep the figure clean
    plot_df = missing_df[missing_df["Total_Missing"] > 0]
    
    if len(plot_df) > 0:
        plt.figure(figsize=(10, 6))
        # Deep blue/gray color palette for professional look
        plt.barh(plot_df["Feature"], plot_df["Percentage_Missing"], color='#2c3e50')
        plt.xlabel("Percentage Missing (%)")
        plt.title("FedLite-Care: Dataset Feature Missingness")
        plt.gca().invert_yaxis()  # Put highest missingness at the top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        os.makedirs("figures", exist_ok=True)
        plot_path = "figures/missingness.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved missingness plot to {plot_path}")
    else:
        print("No missingness detected! Skipping plot.")

if __name__ == "__main__":
    analyze_missingness()
