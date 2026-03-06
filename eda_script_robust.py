import pandas as pd
import numpy as np
import os

# Root directory
root_dir = "/home/gk/github/gk-work/DSML/IM5033-boy-or-girl-2026"
data_path = os.path.join(root_dir, "data/boy-or-girl-2025-train-missingValue.csv")

# Load dataset
df = pd.read_csv(data_path)

# Results log
log_path = os.path.join(root_dir, "eda_detailed_report.txt")

with open(log_path, "w") as f:
    f.write("--- Value Counts for Gender ---\n")
    f.write(df['gender'].value_counts().to_string())
    f.write("\n\n--- Mean Height/Weight by Gender ---\n")
    # Clean non-numeric or extreme values for quick aggregate check
    df_clean = df.copy()
    for col in ['height', 'weight', 'yt']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        # Filter outliers for aggregate check
        if col in ['height', 'weight']:
            df_clean = df_clean[(df_clean[col] > 0) & (df_clean[col] < 300) | df_clean[col].isna()]
    
    f.write(df_clean.groupby('gender')[['height', 'weight']].mean().to_string())
    
    f.write("\n\n--- Missing Values % ---\n")
    f.write((df.isnull().sum() / len(df) * 100).to_string())
    
    f.write("\n\n--- Categorical Feature Unique Values ---\n")
    for col in ['star_sign', 'phone_os']:
        f.write(f"\n{col}:\n")
        f.write(df[col].value_counts().to_string())

print(f"Detailed EDA report saved to {log_path}")
