"""
04_metadata_pipeline.py
=======================
Handles the creation and training of pipelines that combine text features (TF-IDF)
and numerical metadata features (length, count, hour, etc.) via ColumnTransformer.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

def make_preprocessor() -> ColumnTransformer:
    """Return a fresh ColumnTransformer to process text and numeric metadata."""
    return ColumnTransformer(transformers=[
        ("text", TfidfVectorizer(max_features=10000, stop_words="english",
                                 ngram_range=(1, 2), min_df=5, max_df=0.8),
         "cleaned_text"), # Applies to 'cleaned_text' column
        ("num", MinMaxScaler(),
         ["text_length", "word_count", "special_char_count", "hour", "is_weekend"]),
    ])

def build_metadata_models() -> dict:
    """Returns a dictionary of un-trained Scikit-Learn pipelines combining features."""
    return {
        "Naive Bayes": Pipeline([
            ("prep", make_preprocessor()), 
            ("clf", MultinomialNB())
        ]),
        "SVM (LinearSVC)": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=10000)))
        ]),
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    }

def train_metadata_models(models: dict, X_train, y_train) -> dict:
    """Train all models. X_train must be a DataFrame containing text and metadata."""
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name} with metadata...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def save_model(model: Pipeline, le, pipeline_path: str, encoder_path: str):
    """Save the model and label encoder via joblib."""
    joblib.dump(model, pipeline_path, compress=3)
    joblib.dump(le, encoder_path)
    print(f"Pipeline saved to: {pipeline_path}")
    print(f"Encoder saved to: {encoder_path}")

if __name__ == "__main__":
    print("Metadata pipeline module ready.")
