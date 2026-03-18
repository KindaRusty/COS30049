"""
export_training_data.py
========================
One-time script to generate the final processed training dataset CSV.
Outputs: training_dataset_final.csv with columns used by the model.
"""

import os
import sys

# Add parent to path so src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import (
    load_combined_datasets,
    clean_dataframe,
    clean_text,
    extract_metadata,
    encode_labels
)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    original_path = os.path.join(BASE_DIR, "Spam-50k.csv")
    new_dataset_path = os.path.join(BASE_DIR, "spam_Emails_data.csv")
    output_path = os.path.join(BASE_DIR, "training_dataset_final.csv")

    print("-" * 60)
    print("STEP 1: Loading and merging datasets...")
    print("-" * 60)
    df = load_combined_datasets(original_path, new_dataset_path)

    print("\nSTEP 2: Cleaning dataframe...")
    df = clean_dataframe(df)

    print("STEP 3: Applying text cleaning (this may take a few minutes)...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

    print("STEP 4: Extracting metadata features...")
    df = extract_metadata(df)

    print("STEP 5: Encoding labels...")
    df, le = encode_labels(df)

    # Select only the columns used for model training
    # Explicitly exclude: label, hour, is_weekend (internal training features, not for export)
    export_cols = [
        "Spam/Ham",
        "cleaned_text",
        "text_length",
        "word_count",
        "special_char_count"
    ]
    cols_to_drop = ["label", "hour", "is_weekend"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df_export = df[export_cols].copy()

    print(f"\nSTEP 6: Exporting to {output_path}...")
    df_export.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print(f"  Rows: {len(df_export)}")
    print(f"  Columns: {list(df_export.columns)}")
    print(f"  File: {output_path}")
    print(f"  Label distribution:")
    print(df_export['Spam/Ham'].value_counts().to_string())
    print("=" * 60)

if __name__ == "__main__":
    main()