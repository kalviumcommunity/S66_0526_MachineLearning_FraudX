"""
eda.py

Responsible for:
- Systematically inspecting feature distributions
- Generating histograms and boxplots for numerical features
- Analyzing categorical feature frequencies
- Comparing feature distributions across target classes
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import RAW_DATA_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, REPORTS_DIR

def run_eda():
    """
    Execute full Exploratory Data Analysis workflow.
    """
    print("Starting Feature Distribution Inspection...\n")
    
    # 1. Load Data
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Create reports directory for plots if it doesn't exist
    plots_dir = os.path.join(REPORTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # --- Numerical Feature Inspection ---
    print("--- Numerical Feature Inspection ---")
    summary_stats = df[NUMERICAL_FEATURES].describe()
    print(summary_stats)
    
    skewness = df[NUMERICAL_FEATURES].skew()
    print("\nSkewness:")
    print(skewness)
    
    # Plot Histograms
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(NUMERICAL_FEATURES, 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "histograms.png"))
    plt.close()
    
    # Plot Boxplots (for outlier detection)
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(NUMERICAL_FEATURES, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "boxplots.png"))
    plt.close()
    
    # --- Categorical Feature Inspection ---
    print("\n--- Categorical Feature Inspection ---")
    for col in CATEGORICAL_FEATURES:
        print(f"\nValue Counts for {col}:")
        print(df[col].value_counts())
        print(f"Unique levels: {df[col].nunique()}")
        
    # --- Target-Based Comparison ---
    print("\n--- Target-Based Comparison ---")
    # Compare numerical features across target classes
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(NUMERICAL_FEATURES, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x=TARGET_COLUMN, y=col, data=df)
        plt.title(f"{col} vs {TARGET_COLUMN}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "target_comparison.png"))
    plt.close()
    
    print(f"\nEDA Complete. Plots saved to: {plots_dir}")

if __name__ == "__main__":
    run_eda()
