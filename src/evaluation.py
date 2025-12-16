"""
Evaluation utilities for classification models.
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Print accuracy, classification report, and confusion matrix."""
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

def plot_roc(model, X_test, y_test, name="Model"):
    """Plot ROC curve if the model supports probability predictions."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name)
        plt.show()
    else:
        print(f"{name} does not support probability predictions (cannot plot ROC).")