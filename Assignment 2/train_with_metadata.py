"""
train_with_metadata.py
======================
Train spam classifiers using both text (TF-IDF) and metadata features
(text_length, word_count, special_char_count, hour, is_weekend).

Saves: saved_models/spam_pipeline_with_metadata.pkl
       saved_models/label_encoder_metadata.pkl
       results_table_with_metadata.png
"""

import os
import re
import sys
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("stopwords",  quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
from sklearn.model_selection import PredefinedSplit

import tkinter as tk
from tkinter import filedialog
import joblib

stop_words = set(stopwords.words("english"))


# 1. Load data
print("1. Selecting Dataset...")
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
target_file = filedialog.askopenfilename(
    title="Upload/Select Training Dataset (Spam-50k.csv)",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)

if not target_file:
    print("No file selected. Exiting.")
    sys.exit()

print(f"Loading {os.path.basename(target_file)}...")
try:
    df = pd.read_csv(target_file, low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(target_file, encoding="latin1", low_memory=False)

# 2. Data cleaning and preprocessing
cols_to_drop = [c for c in df.columns if c.startswith("Unnamed") or c == "Message ID"]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

if "Spam/Ham" not in df.columns:
    print("Column 'Spam/Ham' not found. Exiting.")
    sys.exit()

df["Spam/Ham"] = df["Spam/Ham"].astype(str).str.lower().str.strip()
df = df[df["Spam/Ham"].isin(["ham", "spam"])].copy()

# 3. Preprocessing: Combine Subject + Message, clean text, and extract metadata features
df["Subject"] = df["Subject"].fillna("")
df["Message"] = df["Message"].fillna("")
df["combined_text"] = df["Subject"].astype(str) + " " + df["Message"].astype(str)
df = df[df["combined_text"].str.strip() != ""]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "url_placeholder",   text)
    text = re.sub(r"\S+@\S+",                 "email_placeholder",  text)
    text = re.sub(r"\d+",                     "number_placeholder", text)
    text = re.sub(r"[^\w\s!$?\-€£]", "",      text)
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in stop_words and len(w) > 1])


print("2. Cleaning text...")
df["cleaned_text"] = df["combined_text"].apply(clean_text)

# 4. Metadata feature extraction
print("3. Extracting Metadata Features...")
df["text_length"]       = df["combined_text"].apply(lambda x: len(str(x)))
df["word_count"]        = df["combined_text"].apply(lambda x: len(str(x).split()))
df["special_char_count"]= df["combined_text"].apply(
    lambda x: sum(1 for c in str(x) if c in string.punctuation)
)

if "Date" in df.columns:
    df["parsed_date"]  = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df["hour"]         = df["parsed_date"].dt.hour.fillna(12)
    df["is_weekend"]   = df["parsed_date"].dt.dayofweek.isin([5, 6]).astype(int)
    df.drop(columns=["Date", "parsed_date"], inplace=True, errors="ignore")
else:
    df["hour"]       = 12
    df["is_weekend"] = 0

le = LabelEncoder()
df["label"] = le.fit_transform(df["Spam/Ham"])

# 5. Train-test split
FEATURES = ["cleaned_text", "text_length", "word_count",
            "special_char_count", "hour", "is_weekend"]
X = df[FEATURES]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# 6. Pipeline setup with ColumnTransformer
def make_preprocessor():
    """Return a fresh ColumnTransformer to avoid state sharing."""
    return ColumnTransformer(transformers=[
        ("text", TfidfVectorizer(max_features=10000, stop_words="english",
                                 ngram_range=(1, 2), min_df=5, max_df=0.8),
         "cleaned_text"),
        ("num", MinMaxScaler(),
         ["text_length", "word_count", "special_char_count", "hour", "is_weekend"]),
    ])


models = {
    "Naive Bayes":        Pipeline([("prep", make_preprocessor()), ("clf", MultinomialNB())]),
    "SVM (LinearSVC)":    Pipeline([("prep", make_preprocessor()),
                                    ("clf", CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=10000)))]),
    "Logistic Regression":Pipeline([("prep", make_preprocessor()),
                                    ("clf", LogisticRegression(random_state=42, max_iter=1000))]),
    "Random Forest":      Pipeline([("prep", make_preprocessor()),
                                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]),
}

# 7. Train models and evaluate on test set
print("\n4. Training Models with Metadata...")
results_data = []
best_f1, best_model, best_model_name = 0, None, ""

for name, model in models.items():
    print(f"   -> Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test,  y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test,    y_pred, zero_division=0)
    f1   = f1_score(y_test,        y_pred, zero_division=0)

    if f1 > best_f1:
        best_f1, best_model, best_model_name = f1, model, name

    results_data.append({
        "Model":          name,
        "Accuracy (%)":   round(acc  * 100, 2),
        "Precision (%)":  round(prec * 100, 2),
        "Recall (%)":     round(rec  * 100, 2),
        "F1-Score (%)":   round(f1   * 100, 2),
    })

# 8. Hyperparameter tuning with GridSearchCV and PredefinedSplit
print(f"\n5. GridSearchCV with PredefinedSplit for '{best_model_name}'...")

param_grids = {
    "Naive Bayes":         {"clf__alpha": [0.01, 0.1, 0.5, 1.0, 5.0]},
    "SVM (LinearSVC)":     {"clf__estimator__C": [0.1, 1.0, 10.0]},
    "Logistic Regression": {"clf__C": [0.1, 1.0, 10.0]},
    "Random Forest":       {"clf__n_estimators": [50, 100, 200]},
}

# Combine train+val; validation indices = 0, training indices = -1
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
test_fold   = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
ps          = PredefinedSplit(test_fold)

gs = GridSearchCV(
    estimator=models[best_model_name],
    param_grid=param_grids[best_model_name],
    cv=ps, scoring="f1", n_jobs=-1, verbose=0, refit=False,
)
gs.fit(X_train_val, y_train_val)

print(f"Best Params : {gs.best_params_}")
print(f"Best Val F1 : {gs.best_score_:.4f}")

# Refit tuned model on training set only
tuned_model = models[best_model_name].set_params(**gs.best_params_)
tuned_model.fit(X_train, y_train)

# 9. Final evaluation of tuned model on test set
results_df = pd.DataFrame(results_data)
print("\nFINAL RESULTS (Test Set)")
print(results_df.to_string(index=False))

y_pred_tuned = tuned_model.predict(X_test)
print(f"\nTUNED {best_model_name.upper()} - DETAILED METRICS")
print(f"Accuracy : {accuracy_score(y_test,  y_pred_tuned) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred_tuned, zero_division=0) * 100:.2f}%")
print(f"Recall   : {recall_score(y_test,    y_pred_tuned, zero_division=0) * 100:.2f}%")
print(f"F1-Score : {f1_score(y_test,        y_pred_tuned, zero_division=0) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=le.classes_))

# 10. Save results table as image
print("\n6.Saving results table image...")
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("tight"); ax.axis("off")
table = ax.table(cellText=results_df.values, colLabels=results_df.columns,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

for (row, col), cell in table._cells.items():
    if row == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#4CAF50")
    else:
        cell.set_facecolor("#F9F9F9")
        if col == 0:
            cell.set_text_props(weight="bold")

plt.title("Model Performance with Metadata Features",
          fontweight="bold", pad=20, size=14)
plt.savefig("results_table_with_metadata.png", bbox_inches="tight", dpi=300)
plt.close()

# 11. Save best tuned model and label encoder
os.makedirs("saved_models", exist_ok=True)
joblib.dump(tuned_model, "saved_models/spam_pipeline_with_metadata.pkl")
joblib.dump(le,          "saved_models/label_encoder_metadata.pkl")
print(f"\n=> Best tuned model ({best_model_name}) saved to 'saved_models/'")
print("spam_pipeline_with_metadata.pkl")
print("label_encoder_metadata.pkl")
