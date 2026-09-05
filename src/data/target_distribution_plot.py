import matplotlib.pyplot as plt
import json
import os

def plot_target_distribution():
    json_path = "results/target_distribution.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Target distribution file not found at {json_path}")
        
    with open(json_path, "r") as f:
        dist_data = json.load(f)
        
    binary_dist = dist_data["binary_30day_readmitted"]
    
    classes = ["Not Readmitted\nor >30 Days (0)", "Readmitted\nwithin 30 Days (1)"]
    counts = [binary_dist["0"], binary_dist["1"]]
    percentages = [c / sum(counts) * 100 for c in counts]
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(classes, counts, color=['#34495e', '#e74c3c'], width=0.5)
    
    # Add counts and percentages on top of the bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1000, 
                 f"{count:,}\n({pct:.2f}%)", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.ylabel("Encounter Count")
    plt.title("FedLite-Care: Binary 30-Day Readmission Target Distribution")
    plt.ylim(0, max(counts) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plot_path = "figures/target_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved target distribution plot to {plot_path}")

if __name__ == "__main__":
    plot_target_distribution()
