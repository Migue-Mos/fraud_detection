"""
Main script for Fraud Detection pipeline.
Loads data, preprocesses it, trains models, evaluates them, and saves the best one.
"""

import os
import joblib
import matplotlib.pyplot as plt
from src.data_loader import load_data, basic_info
from src.preprocessing import clean_data, split_features, scale_numeric
from src.modeling import train_logistic, train_random_forest, train_svm
from src.evaluation import evaluate_model, plot_roc
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay

# --- 1 Load data ---
data_path = "data/fraud_data.csv"
df = load_data("data/raw/creditcard.csv")
basic_info(df)

# --- 2 Clean data ---
df = clean_data(df)

# --- 3 Split features and target ---
X, y = split_features(df, target_col="Class")
print(X.shape, y.shape)
print(y.value_counts())

# --- 4 Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)

# --- 5 Scale numeric features ---
numeric_cols = X.select_dtypes(include="number").columns.tolist()
X_train_scaled, X_test_scaled, scaler = scale_numeric(X_train.copy(), X_test.copy(), numeric_cols)

# --- 6 Train models ---
print("training logistic regression...")
log_model = train_logistic(X_train_scaled, y_train)
print("training random forest...")
rf_model = train_random_forest(X_train_scaled, y_train)
print("training SVM...")
svm_model = train_svm(X_train_scaled, y_train)

# --- 7 Evaluate models ---
print("\n=== Logistic Regression ===")
evaluate_model(log_model, X_test_scaled, y_test)

print("\n=== Random Forest ===")
evaluate_model(rf_model, X_test_scaled, y_test)

print("\n=== SVM ===")
evaluate_model(svm_model, X_test_scaled, y_test)

# --- 8 Plot ROC curves together ---
os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Logistic Regression ROC
RocCurveDisplay.from_predictions(
    y_test, log_model.predict_proba(X_test_scaled)[:,1],
    ax=axes[0], name="Logistic Regression"
)
# Random Forest ROC
RocCurveDisplay.from_predictions(
    y_test, rf_model.predict_proba(X_test_scaled)[:,1],
    ax=axes[1], name="Random Forest"
)
# SVM ROC
RocCurveDisplay.from_predictions(
    y_test, svm_model.predict_proba(X_test_scaled)[:,1],
    ax=axes[2], name="SVM"
)
plt.tight_layout()
plt.savefig("results/roc_comparison.png")  # Save the figure
plt.close()  # Close the figure to avoid opening a window
print("ROC curves saved in 'results/roc_comparison.png'")


# --- 9 Save models and scaler ---
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/best_model_rf.pkl")
joblib.dump(log_model, "models/logistic_model.pkl")
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Models and scaler saved in 'models/'")