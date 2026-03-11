"""
evaluate_robustness.py
======================
Stress-test the trained spam pipeline (spam_pipeline_with_metadata.pkl)
against augmented / synthetic data from synthetic_nlpaug.csv.

Outputs:
  - Console: Accuracy, Precision, Recall, F1-Score + Classification Report
  - robustness_confusion_matrix.png
"""

import re
import string

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

stop_words = set(stopwords.words("english"))


# 1.Loading model and augmented data
print("1. Loading model and data...")
try:
    model = joblib.load("saved_models/spam_pipeline_with_metadata.pkl")
    le    = joblib.load("saved_models/label_encoder_metadata.pkl")
except FileNotFoundError:
    print("ERROR: Model files not found in 'saved_models/'.")
    print("       Please run train_with_metadata.py first.")
    raise SystemExit(1)
try:
    df_aug = pd.read_csv("synthetic_nlpaug.csv")
except FileNotFoundError:
    print("ERROR: 'synthetic_nlpaug.csv' not found.")
    print("       Please run gen_nlpaug.py first.")
    raise SystemExit(1)
print(f"-> Loaded {len(df_aug):,} augmented rows.")
print(f"   Class distribution: {df_aug['Spam/Ham'].value_counts().to_dict()}")


# 2. Feature engineering to match training schema
print("\n2. Feature engineering to match training schema...")

df_aug["Subject"] = df_aug["Subject"].fillna("")
df_aug["Message"] = df_aug["Message"].fillna("")
df_aug["combined_text"] = (
    df_aug["Subject"].astype(str) + " " + df_aug["Message"].astype(str)
)

def clean_text(text: str) -> str:
    """Identical pipeline to train_with_metadata.py."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+","url_placeholder",text)
    text = re.sub(r"\S+@\S+","email_placeholder",text)
    text = re.sub(r"\d+","number_placeholder",text)
    text = re.sub(r"[^\w\s!$?\-€£]","",text)
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in stop_words and len(w) > 1])

df_aug["cleaned_text"]      = df_aug["combined_text"].apply(clean_text)
df_aug["text_length"]       = df_aug["combined_text"].apply(lambda x: len(str(x)))
df_aug["word_count"]        = df_aug["combined_text"].apply(lambda x: len(str(x).split()))
df_aug["special_char_count"]= df_aug["combined_text"].apply(
    lambda x: sum(1 for c in str(x) if c in string.punctuation)
)

# Simulate metadata features (hour and is_weekend) with random values
np.random.seed(42)
df_aug["hour"] = np.random.choice([2, 3, 14, 15], size=len(df_aug))
df_aug["is_weekend"] = np.random.choice([0, 1], p=[0.7, 0.3], size=len(df_aug))

# Encoding true labels using the same LabelEncoder as training
df_aug["true_label"] = le.transform(df_aug["Spam/Ham"])

X_test = df_aug[["cleaned_text", "text_length", "word_count",
                  "special_char_count", "hour", "is_weekend"]]
y_true = df_aug["true_label"]


# 3. Running robustness evaluation
print("\n3. Running robustness evaluation on augmented data...")
y_pred = model.predict(X_test)


# 4. Metrics and classification report
acc = accuracy_score(y_true,  y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true,    y_pred, zero_division=0)
f1 = f1_score(y_true,        y_pred, zero_division=0)

print("\n" + "=" * 60)
print("Robustness Result (Augmented / Synthetic Data)")
print("=" * 60)
print(f"  Accuracy  : {acc  * 100:.2f}%")
print(f"  Precision : {prec * 100:.2f}%")
print(f"  Recall    : {rec  * 100:.2f}%")
print(f"  F1-Score  : {f1   * 100:.2f}%")
print("-" * 60)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))


# 5. Confusion matrix visualization
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Augmented Data Robustness Test",
          fontweight="bold", pad=12)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("robustness_confusion_matrix.png", dpi=300)
plt.close()

print("=> Saved 'robustness_confusion_matrix.png'")
