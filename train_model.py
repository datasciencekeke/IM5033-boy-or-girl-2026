import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from data_pipeline import clean_data, get_preprocessing_pipeline

# Root directory
root_dir = "/home/gk/github/gk-work/DSML/IM5033-boy-or-girl-2026"
data_path = os.path.join(root_dir, "data/boy-or-girl-2025-train-missingValue.csv")

def train():
    # Load and clean data
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    X = df.drop('gender', axis=1)
    y = df['gender'].map({1: 0, 2: 1}) # Mapping 1 to 0 (Male), 2 to 1 (Female) for models
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = get_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1])), 
                                 use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_score = 0
    results = {}

    print("--- Model Evaluation ---")
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_val_processed)
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, model.predict_proba(X_val_processed)[:, 1])
        
        results[name] = {'Accuracy': acc, 'ROC-AUC': auc}
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(classification_report(y_val, y_pred))
        
        if auc > best_score:
            best_score = auc
            best_model = (name, model)

    print(f"\nBest Model: {best_model[0]} with ROC-AUC {best_score:.4f}")
    
    # Save the best model and preprocessor
    model_save_path = os.path.join(root_dir, "best_model.joblib")
    preprocessor_save_path = os.path.join(root_dir, "preprocessor.joblib")
    
    joblib.dump(best_model[1], model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)
    
    print(f"\nModel saved to {model_save_path}")
    print(f"Preprocessor saved to {preprocessor_save_path}")

if __name__ == "__main__":
    train()
