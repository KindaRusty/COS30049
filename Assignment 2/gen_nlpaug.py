"""
gen_nlpaug.py
=============
Generate augmented spam/ham rows using nlpaug to test model robustness.
Strategies: synonym replacement, random word deletion, character swap.
Output: synthetic_nlpaug.csv (~1000 balanced rows)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import random
import pandas as pd
import nltk
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

# Download necessary NLTK resources for augmentation
for pkg in ("wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
            "stopwords", "punkt", "punkt_tab", "omw-1.4"):
    nltk.download(pkg, quiet=True)

SEED         = 42
N_PER_CLASS  = 250
N_TARGET     = 1000
random.seed(SEED)

# 1. Load seed data from Spam-50k.csv
print("Loading Spam-50k.csv...")
usecols = ["Message ID", "Subject", "Message", "Spam/Ham"]
raw = pd.read_csv("Spam-50k.csv", usecols=usecols)
raw["Spam/Ham"] = raw["Spam/Ham"].astype(str).str.lower().str.strip()
raw = raw[raw["Spam/Ham"].isin(["ham", "spam"])].dropna(subset=["Message"])
raw["Subject"] = raw["Subject"].fillna("(no subject)")
raw["Message"] = raw["Message"].str[:500]

spam_seed = raw[raw["Spam/Ham"] == "spam"].sample(
    n=min(N_PER_CLASS, len(raw[raw["Spam/Ham"] == "spam"])), random_state=SEED
).reset_index(drop=True)
ham_seed = raw[raw["Spam/Ham"] == "ham"].sample(
    n=min(N_PER_CLASS, len(raw[raw["Spam/Ham"] == "ham"])), random_state=SEED
).reset_index(drop=True)

seed_df = pd.concat([spam_seed, ham_seed], ignore_index=True)
print(f"Seed data: {seed_df.shape} ({seed_df['Spam/Ham'].value_counts().to_dict()})")

# 2. Initialize augmenters
print("Initializing augmenters...")
aug_synonym = naw.SynonymAug(aug_src="wordnet", aug_p=0.2)
aug_delete  = naw.RandomWordAug(action="delete", aug_p=0.15)
aug_char    = nac.RandomCharAug(action="swap", aug_char_p=0.05)

AUGMENTERS = [aug_synonym, aug_delete, aug_char]
AUG_NAMES  = ["synonym", "delete", "char_swap"]


def safe_augment(augmenter, text):
    """Safely apply augmentation; return None if failed or unmodified."""
    try:
        text = str(text).strip()
        if len(text) < 5:
            return None
        result = augmenter.augment(text)
        if isinstance(result, list):
            result = result[0]
        result = str(result).strip()
        return result if (result and result != text) else None
    except Exception:
        return None


# 3. Generate augmented rows
print(f"Augmenting {len(seed_df)} rows...")
rows = []
for i, (_, row) in enumerate(seed_df.iterrows()):
    if i % 100 == 0:
        print(f"  {i}/{len(seed_df)}...")
    label, subject, msg = row["Spam/Ham"], str(row["Subject"]), str(row["Message"])

    for aug, aug_name in zip(AUGMENTERS, AUG_NAMES):
        aug_message = safe_augment(aug, msg)
        if aug_message:
            aug_subject = safe_augment(aug, subject) or subject
            rows.append({
                "Strategy": aug_name,
                "Subject":  aug_subject,
                "Message":  aug_message,
                "Spam/Ham": label,
            })

aug_df = pd.DataFrame(rows)

# 4. Balance classes and limit to target size
spam_aug = aug_df[aug_df["Spam/Ham"] == "spam"]
ham_aug  = aug_df[aug_df["Spam/Ham"] == "ham"]
n_each   = min(len(spam_aug), len(ham_aug), N_TARGET // 2)

final_df = pd.concat([
    spam_aug.sample(n=n_each, random_state=SEED),
    ham_aug.sample(n=n_each, random_state=SEED)
], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

# 5. Write to CSV
final_df.insert(0, "Message ID", range(len(final_df)))
final_df[["Message ID", "Subject", "Message", "Spam/Ham", "Strategy"]].to_csv(
    "synthetic_nlpaug.csv", index=False, encoding="utf-8"
)

print(f"\nWritten synthetic_nlpaug.csv: {final_df.shape}")
print("\nClass distribution:")
print(final_df["Spam/Ham"].value_counts().to_string())
print("\nAugmentation strategy breakdown:")
print(final_df["Strategy"].value_counts().to_string())
