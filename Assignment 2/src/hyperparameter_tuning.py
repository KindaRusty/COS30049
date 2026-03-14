"""
hyperparameter_tuning.py
===========================
Handles GridSearchCV for hyperparameter tuning using a predefined validation split 
to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

def get_param_grids() -> dict:
    """Returns hyperparameter search grids for each model type to be used in GridSearchCV."""
    return {
        # Naive Bayes: Alpha controls additive smoothing (Lidstone smoothing)
        "Naive Bayes": {
            "clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        },
        # SVM (LinearSVC): C is the regularization parameter (inverse of regularization strength)
        "SVM (LinearSVC)": {
            "clf__estimator__C": [0.01, 0.1, 1.0, 10.0],
            "clf__estimator__max_iter": [5000, 10000]
        },
        # Logistic Regression: C is inversely proportional to regularization; Solver selection
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__solver": ["lbfgs", "liblinear"]
        },
        # Random Forest: Number of trees, maximum depth, and minimum samples per split
        "Random Forest": {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5]
        }
    }

def run_gridsearch(model, param_grid, X_train, X_val, y_train, y_val, n_jobs=-1) -> tuple:
    """
    Run GridSearchCV using a PredefinedSplit combining X_train and X_val.
    Ensures validation set is strictly used for validation during search, not training.
    """
    # Combine train and val sets for the GridSearchCV fit call
    if isinstance(X_train, pd.Series):
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
    else:  # DataFrame for metadata pipelines
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)
        
    # Create test_fold mask: -1 means training sample, 0 means validation sample
    # This prevents the primary training data from being used in the validation folds
    test_fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
    ps = PredefinedSplit(test_fold)
    
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=ps,
        scoring="f1", # Optimizing based on f1 score due to potentially imbalanced context
        n_jobs=n_jobs,
        verbose=1,
        refit=False # Emphasize that we manually refit on train only later to maintain experimental integrity
    )
    
    print(f"Starting GridSearchCV...")
    gs.fit(X_train_val, y_train_val)
    return gs.best_params_, gs.best_score_

def refit_and_evaluate(model_pipeline_class, best_params, X_train, X_test, y_train, y_test, le=None):
    """
    Refit the model on purely X_train using best_params, then evaluate on X_test.
    """
    tuned_model = model_pipeline_class.set_params(**best_params)
    tuned_model.fit(X_train, y_train)
    
    y_pred = tuned_model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(y_test, y_pred, target_names=le.classes_ if le else None)
    }
    return tuned_model, metrics

if __name__ == "__main__":
    print("Hyperparameter tuning module ready.")
