"""
clustering_augmentation.py
=============================
Handles K-Means clustering of spam topics, NLP augmentation (synthetic data generation),
and evaluating robustness on augmented data.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore")

def run_kmeans(df_spam: pd.DataFrame, tfidf_matrix, n_clusters=4, random_seed=42) -> Tuple:
    """Run K-Means clustering on the spam TF-IDF matrix."""
    # Initialize KMeans with 4 clusters as identified during EDA (phishing, pharmacy, piracy, branding)
    km = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10, max_iter=300)
    km.fit(tfidf_matrix)
    
    df_spam_clustered = df_spam.copy()
    df_spam_clustered["cluster"] = km.labels_
    
    # Calculate silhouette score to evaluate clustering quality
    sil_score = silhouette_score(tfidf_matrix, km.labels_)
    return km, df_spam_clustered, sil_score

def get_top_keywords(km: KMeans, tfidf_vectorizer, n=10) -> dict:
    """Retrieve top n keywords per cluster based on centroid distance."""
    terms = tfidf_vectorizer.get_feature_names_out()
    centroids = km.cluster_centers_
    top_keywords = {}
    
    for i in range(km.n_clusters):
        # Sort indices of the centroid vector in descending order to find most prominent terms
        top_idx = centroids[i].argsort()[::-1][:n]
        top_kws = [terms[j] for j in top_idx]
        top_keywords[f"Cluster {i}"] = top_kws
        
    return top_keywords

def safe_augment(augmenter, text) -> Optional[str]:
    """Safely apply augmentation; return None if failed or result is identical to input."""
    try:
        text = str(text).strip()
        # Avoid augmenting very short strings that lack semantic context
        if len(text) < 5:
            return None
        result = augmenter.augment(text)
        if isinstance(result, list):
            result = result[0]
        result = str(result).strip()
        
        # Return result only if it's non-empty and has actually been changed by the augmenter
        return result if (result and result != text) else None
    except Exception:
        return None

def generate_augmented_data(df: pd.DataFrame, n_per_class=250, n_target=1000, seed=42) -> pd.DataFrame:
    """Generate NLP augmented data (synonym, random deletion, character swap)."""
    # 1. Prepare Seed Data
    spam_seed = df[df["Spam/Ham"] == "spam"].sample(n=min(n_per_class, len(df[df["Spam/Ham"] == "spam"])), random_state=seed)
    ham_seed = df[df["Spam/Ham"] == "ham"].sample(n=min(n_per_class, len(df[df["Spam/Ham"] == "ham"])), random_state=seed)
    seed_df = pd.concat([spam_seed, ham_seed], ignore_index=True)
    
    # 2. Augmenters configuration
    # SynonymAug: Replaces words with synonyms from WordNet
    aug_synonym = naw.SynonymAug(aug_src="wordnet", aug_p=0.2)
    # RandomWordAug: Deletes random words to simulate missing data or informal typing
    aug_delete  = naw.RandomWordAug(action="delete", aug_p=0.15)
    # RandomCharAug: Swaps adjacent characters to simulate typos
    aug_char    = nac.RandomCharAug(action="swap", aug_char_p=0.05)
    
    augmenters = [aug_synonym, aug_delete, aug_char]
    aug_names  = ["synonym", "delete", "char_swap"]
    
    rows = []
    print(f"Augmenting {len(seed_df)} rows...")
    for i, (_, row) in enumerate(seed_df.iterrows()):
        label, subject, msg = row["Spam/Ham"], str(row.get("Subject", "")), str(row.get("Message", ""))
        for aug, aug_name in zip(augmenters, aug_names):
            # Apply augmentation to the message body
            aug_message = safe_augment(aug, msg)
            if aug_message:
                # Also try augmenting the subject, or use original if augmentation fails
                aug_subject = safe_augment(aug, subject) or subject
                rows.append({"Strategy": aug_name, "Subject": aug_subject, "Message": aug_message, "Spam/Ham": label})
                
    aug_df = pd.DataFrame(rows)
    
    # 3. Balance and Limit Size
    spam_aug = aug_df[aug_df["Spam/Ham"] == "spam"]
    ham_aug  = aug_df[aug_df["Spam/Ham"] == "ham"]
    n_each   = min(len(spam_aug), len(ham_aug), n_target // 2)
    
    final_df = pd.concat([
        spam_aug.sample(n=n_each, random_state=seed),
        ham_aug.sample(n=n_each, random_state=seed)
    ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    final_df.insert(0, "Message ID", range(len(final_df)))
    return final_df

def run_robustness_test(model, le, aug_df: pd.DataFrame, x_cols: list) -> dict:
    """Test trained pipeline on augmented dataset."""
    X_test = aug_df[x_cols]
    y_true = le.transform(aug_df['Spam/Ham'])
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=le.classes_)
    
    return {"Accuracy": acc, "classification_report": report, "y_true": y_true, "y_pred": y_pred}

if __name__ == "__main__":
    print("Clustering and augmentation module ready.")