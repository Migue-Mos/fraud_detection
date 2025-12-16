"""
Model training utilities: Logistic Regression, Random Forest, SVM.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def train_logistic(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        max_depth=None, 
        n_jobs=-1, )
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train a Support Vector Machine classifier."""
    linear_svc = LinearSVC(max_iter=10000)
    model = CalibratedClassifierCV(linear_svc)  # allow to predict probabilities
    model.fit(X_train, y_train)
    return model
