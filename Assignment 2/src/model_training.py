"""
model_training.py
====================
Handles individual model definition, training, and evaluation for text-only pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def get_tfidf() -> TfidfVectorizer:
    """Returns a fresh instance of TfidfVectorizer with optimized parameters for spam detection."""
    # max_features=10000: limits vocabulary to top 10k terms to manage dimensionality
    # ngram_range=(1, 2): captures unigrams and bigrams for better contextual understanding
    return TfidfVectorizer(max_features=10000, stop_words="english",
                           ngram_range=(1, 2), min_df=5, max_df=0.8)

def build_models() -> dict:
    """Returns a dictionary of un-trained Scikit-Learn pipelines for baseline comparison."""
    return {
        # Multinomial Naive Bayes: Fast and effective for text classification (standard baseline)
        "Naive Bayes": Pipeline([
            ("tfidf", get_tfidf()), 
            ("clf", MultinomialNB())
        ]),
        # LinearSVC: Support Vector Machine optimized for text; Calibrated to provide probability scores
        "SVM (LinearSVC)": Pipeline([
            ("tfidf", get_tfidf()), 
            ("clf", CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=10000)))
        ]),
        # Logistic Regression: Interpretability and strong performance on linearly separable features
        "Logistic Regression": Pipeline([
            ("tfidf", get_tfidf()), 
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        # Random Forest: Ensemble method to capture non-linear relationships and interactions
        "Random Forest": Pipeline([
            ("tfidf", get_tfidf()), 
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    }

def train_all_models(models: dict, X_train, y_train) -> dict:
    """Trains all models provided in the dictionary."""
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_model(model: Pipeline, X_test, y_test, le=None) -> dict:
    """Evaluates a single model and returns metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    
    report = classification_report(y_test, y_pred, target_names=le.classes_ if le else None)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "y_pred": y_pred,
        "classification_report": report
    }

if __name__ == "__main__":
    print("Model training module ready.")
