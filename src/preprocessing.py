"""
Preprocessing utilities: cleaning, encoding, scaling.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and fix types."""
    df = df.dropna()
    return df

def split_features(df: pd.DataFrame, target_col: str):
    """Split into X (features) and y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def scale_numeric(X_train, X_test, numeric_cols):
    """Scale numeric columns using StandardScaler."""
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, scaler
