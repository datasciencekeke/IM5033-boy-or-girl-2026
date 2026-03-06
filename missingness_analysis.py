import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Root directory
root_dir = "/home/gk/github/gk-work/DSML/IM5033-boy-or-girl-2026"
data_path = os.path.join(root_dir, "data/boy-or-girl-2025-train-missingValue.csv")

def audit_missingness(df_path):
    """
    Performs an audit of missing values in the dataset.
    """
    df = pd.read_csv(df_path)
    
    # 1. Initial Audit
    print("--- Initial Audit ---")
    print(df.info())
    
    # 2. Missing Counts and Percentages
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    
    missing_report = pd.DataFrame({
        'Null Count': null_counts,
        'Percentage (%)': null_percentages,
        'Dtype': df.dtypes
    })
    
    print("\n--- Missingness Summary ---")
    print(missing_report[missing_report['Null Count'] > 0].sort_values(by='Null Count', ascending=False))
    
    # 3. Descriptive Stats for Numerical Columns (Baseline)
    print("\n--- Descriptive Statistics (Baseline) ---")
    print(df.describe())
    
    # 4. Missing Value Correlation (Heatmap)
    # Check if there are columns with missing values to avoid errors
    cols_with_missing = missing_report[missing_report['Null Count'] > 0].index.tolist()
    
    if cols_with_missing:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[cols_with_missing].isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Value Heatmap')
        heatmap_path = os.path.join(root_dir, "missing_value_heatmap.png")
        plt.savefig(heatmap_path)
        print(f"\nHeatmap saved to {heatmap_path}")
        
        # Missingness Correlation
        plt.figure(figsize=(10, 8))
        msno_corr = df[cols_with_missing].isnull().corr()
        sns.heatmap(msno_corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Missingness Correlation')
        corr_path = os.path.join(root_dir, "missing_correlation.png")
        plt.savefig(corr_path)
        print(f"Missingness correlation saved to {corr_path}")

    # Output report to text file
    report_path = os.path.join(root_dir, "missingness_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Missing Data Audit Report ---\n\n")
        f.write("Total Records: {}\n".format(len(df)))
        f.write("\nMissing Value Summary:\n")
        f.write(missing_report[missing_report['Null Count'] > 0].sort_values(by='Null Count', ascending=False).to_string())
        f.write("\n\nDescriptive Statistics:\n")
        f.write(df.describe().to_string())
        
    print(f"\nAudit report saved to {report_path}")

if __name__ == "__main__":
    audit_missingness(data_path)
