"""
data_preprocessing.py
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
    """Load the dataset from a CSV file with error handling for file existence and encoding."""
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
        
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        # Fallback for datasets with special characters (common in email data)
        df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
    
    # Drop unnecessary columns
    cols_to_drop = [c for c in df.columns if c.startswith('Unnamed') or c == 'Message ID']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def load_and_standardize_220k_dataset(filepath: str) -> pd.DataFrame:
    """Load and standardize the 220k dataset (label, text) to match the main schema."""
    try:
        df_new = pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        df_new = pd.read_csv(filepath, encoding='latin-1', low_memory=False)

    # Rename columns to match the main pipeline schema
    df_new = df_new.rename(columns={
        "label": "Spam/Ham",
        "text": "Message"
    })

    # Synchronize label values: numeric (0/1) -> string (ham/spam)
    if pd.api.types.is_numeric_dtype(df_new['Spam/Ham']):
        df_new['Spam/Ham'] = df_new['Spam/Ham'].map({0: 'ham', 1: 'spam'})
    else:
        df_new['Spam/Ham'] = df_new['Spam/Ham'].astype(str).str.lower().str.strip()

    # Fill missing columns that the main pipeline expects
    df_new["Subject"] = ""
    df_new["Date"] = pd.NaT

    return df_new[["Spam/Ham", "Subject", "Message", "Date"]]

def compare_datasets(df_main: pd.DataFrame, df_new: pd.DataFrame):
    """Log statistics for both datasets before merging for validation."""
    print("\n" + "="*40)
    print("=== DATASET COMPARISON STATUS ===")
    print("=== Dataset 1 (Spam-50k) ===")
    print(df_main['Spam/Ham'].value_counts())
    print(f"Null messages: {df_main['Message'].isna().sum()}")
    
    print("\n=== Dataset 2 (220k Dataset) ===")
    print(df_new['Spam/Ham'].value_counts())
    print(f"Null messages: {df_new['Message'].isna().sum()}")
    print("="*40 + "\n")

def load_combined_datasets(original_path: str, new_dataset_path: str) -> pd.DataFrame:
    """Load and merge the original dataset with the 220k dataset with deduplication."""
    print("Loading original dataset...")
    df_main = load_dataset(original_path)

    print("Loading 220k dataset...")
    df_new = load_and_standardize_220k_dataset(new_dataset_path)

    # Log statistics before merging
    compare_datasets(df_main, df_new)

    # Concatenate vertically
    print("Merging datasets...")
    df_combined = pd.concat([df_main, df_new], ignore_index=True)

    # Drop rows where Message is empty/NaN
    df_combined = df_combined.dropna(subset=['Message'])

    # IMPORTANT: Remove duplicates after merging from both sources
    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['Message'])
    dedup_count = initial_count - len(df_combined)
    
    print(f"Dropped {dedup_count} duplicates.")
    print(f"Final combined total: {len(df_combined)} rows.")
    
    print("\n=== Combined Label Distribution ===")
    print(df_combined['Spam/Ham'].value_counts())
    print("="*40 + "\n")
    
    return df_combined

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
    
    # regex matches: http/https links or www. prefixes
    text = re.sub(r"http\S+|www\S+|https\S+", "url_placeholder", text)
    
    # regex matches: standard email formats (username@domain)
    text = re.sub(r"\S+@\S+", "email_placeholder", text)
    
    # regex matches: one or more digits
    text = re.sub(r"\d+", "number_placeholder", text)
    
    # regex matches: anything NOT a word character, space, or specific currency/punctuation marks to keep
    # [^\w\s!$?\-€£] -> keep letters, numbers, spaces, and ! $ ? - € £
    text = re.sub(r'[^\w\s!$?\-€£]', '', text)
    
    tokens = word_tokenize(text)
    # Filter out stopwords and very short tokens (noise)
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
    """
    Split into train, validation, and test sets using a two-step process to achieve a 70/15/15 ratio.
    
    Logic:
    1. First split: 70% Train, 30% Temp (using test_size=0.3)
    2. Second split: Temp (30%) is split 50/50 (using val_size=0.5) to get 15% Validation and 15% Test.
    Final Result: 70% Train, 15% Val, 15% Test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    original_path = os.path.join(BASE_DIR, "..", "Spam-50k.csv")
    new_dataset_path = os.path.join(BASE_DIR, "..", "spam_Emails_data.csv")
    
    try:
        df = load_combined_datasets(original_path, new_dataset_path)
        df = clean_dataframe(df)
        df['cleaned_text'] = df['combined_text'].apply(clean_text)
        df = extract_metadata(df)
        df, le = encode_labels(df)
        print("Data preprocessing module ready.")
        print(f"Final dataset shape: {df.shape}")
    except Exception as e:
        print(f"Could not load data for local test: {e}")
