import pandas as pd
import numpy as np
import os
import joblib
from data_pipeline import clean_data

# Root directory
root_dir = "/home/gk/github/gk-work/DSML/IM5033-boy-or-girl-2026"
test_data_path = os.path.join(root_dir, "data/boy-or-girl-2025-test-no-ans-missingValue.csv")
sample_sub_path = os.path.join(root_dir, "data/boy-or-girl-kaggle-sample-submission.csv")

def predict():
    # Load model and preprocessor
    model = joblib.load(os.path.join(root_dir, "best_model.joblib"))
    preprocessor = joblib.load(os.path.join(root_dir, "preprocessor.joblib"))
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    ids = test_df['id']
    
    # Process test data (note: gender is missing or zero/null in test set)
    X_test = clean_data(test_df)
    X_test = X_test.drop('gender', axis=1, errors='ignore')
    
    # Transform
    X_test_processed = preprocessor.transform(X_test)
    
    # Predict
    preds = model.predict(X_test_processed)
    # Map back to original gender labels (0 -> 1, 1 -> 2)
    final_preds = [1 if p == 0 else 2 for p in preds]
    
    # Create submission
    submission = pd.DataFrame({
        'id': ids,
        'gender': final_preds
    })
    
    output_path = os.path.join(root_dir, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    predict()
