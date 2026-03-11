"""
01_data_preprocessing.py
========================
Handles data loading, cleaning, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Ensure required NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOP_WORDS = set(stopwords.words("english"))

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
    
    # Drop unnecessary columns
    cols_to_drop = [c for c in df.columns if c.startswith('Unnamed') or c == 'Message ID']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def clean_dataframe(df: pd.DataFrame, label_col: str = "Spam/Ham") -> pd.DataFrame:
    """Pre-clean dataframe: fill NaNs, combine Subject+Message, filter valid labels."""
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df[label_col].isin(['ham', 'spam'])].copy()
    
    df['Subject'] = df['Subject'].fillna('')
    df['Message'] = df['Message'].fillna('')
    df['combined_text'] = df['Subject'].astype(str) + " " + df['Message'].astype(str)
    
    # Remove rows where combined text is empty
    df = df[df['combined_text'].str.strip() != ""]
    df.drop_duplicates(subset=['combined_text'], inplace=True)
    return df

def clean_text(text: str) -> str:
    """Normalize text: lowercase, replace URLs/emails/numbers, remove punctuation except specific chars."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "url_placeholder", text)
    text = re.sub(r"\S+@\S+", "email_placeholder", text)
    text = re.sub(r"\d+", "number_placeholder", text)
    text = re.sub(r'[^\w\s!$?\-€£]', '', text)
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

def extract_metadata(df: pd.DataFrame, text_col: str = "combined_text") -> pd.DataFrame:
    """Extract metadata features like text length, word count, special char count, hour, and is_weekend."""
    df["text_length"] = df[text_col].apply(lambda x: len(str(x)))
    df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()))
    df["special_char_count"] = df[text_col].apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
    
    if "Date" in df.columns:
        df["parsed_date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
        df["hour"] = df["parsed_date"].dt.hour.fillna(12)
        df["is_weekend"] = df["parsed_date"].dt.dayofweek.isin([5, 6]).astype(int)
        df.drop(columns=["Date", "parsed_date"], inplace=True, errors="ignore")
    else:
        df["hour"] = 12
        df["is_weekend"] = 0
        
    return df

def encode_labels(df: pd.DataFrame, label_col: str = "Spam/Ham"):
    """Encode target labels (ham -> 0, spam -> 1)."""
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[label_col])
    return df, le

def split_dataset(X, y, test_size=0.3, val_size=0.5, random_state=42):
    """Split into train, validation, and test sets.
    test_size here applies to the first split (train/temp).
    val_size applies to the split of temp into val/test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Example usage
    filepath = "../Spam-50k.csv"
    try:
        df = load_dataset(filepath)
        df = clean_dataframe(df)
        df['cleaned_text'] = df['combined_text'].apply(clean_text)
        df = extract_metadata(df)
        df, le = encode_labels(df)
        print("Data preprocessing module ready.")
    except Exception as e:
        print(f"Could not load data for local test: {e}")
