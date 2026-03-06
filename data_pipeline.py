import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_data(df):
    """
    Cleans raw data by handling extreme outliers and specific string errors.
    """
    df = df.copy()
    
    # List of numeric columns
    numeric_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']
    
    for col in numeric_cols:
        # Convert to numeric, turn errors like #NUM! or non-numeric strings to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Replace extreme outliers/infinity with NaN before scaling
        # float32 max is ~3.4e38, so we use a much safer threshold
        df.loc[df[col] > 1e12, col] = np.nan
        df.loc[df[col] < -1e12, col] = np.nan
        
        # Specific business logic outlier handling
        if col == 'height':
            df.loc[(df[col] < 100) | (df[col] > 250), col] = np.nan
        elif col == 'weight':
            df.loc[(df[col] < 30) | (df[col] > 200), col] = np.nan
        elif col == 'yt':
            df.loc[df[col] < 0, col] = np.nan
            
    # Feature Engineering: length of self-intro
    df['self_intro_len'] = df['self_intro'].fillna('').astype(str).apply(len)
    
    return df

def get_preprocessing_pipeline():
    """
    Returns a scikit-learn ColumnTransformer for preprocessing.
    """
    numeric_features = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'self_intro_len']
    categorical_features = ['star_sign', 'phone_os']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

if __name__ == "__main__":
    # Test script locally
    root_dir = "/home/gk/github/gk-work/DSML/IM5033-boy-or-girl-2026"
    data_path = os.path.join(root_dir, "data/boy-or-girl-2025-train-missingValue.csv")
    df = pd.read_csv(data_path)
    
    cleaned_df = clean_data(df)
    print("Cleaned data shape:", cleaned_df.shape)
    print("Missing values in cleaned data:\n", cleaned_df.isnull().sum())
    
    preprocessor = get_preprocessing_pipeline()
    # Note: Target 'gender' should be handled separately
    X = cleaned_df.drop('gender', axis=1)
    y = cleaned_df['gender']
    
    X_processed = preprocessor.fit_transform(X)
    print("Preprocessed features shape:", X_processed.shape)
