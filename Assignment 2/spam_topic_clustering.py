"""
spam_topic_clustering.py
========================
Unsupervised K-Means Clustering on spam-only emails to discover latent
spam campaign topics (e.g. Phishing, Financial Fraud, Ad Spam, Malware).

ML Type : Clustering (K-Means)
Dataset : Spam-50k.csv  (only rows where Spam/Ham == 'spam')
Outputs : spam_clusters_visualization.png
"""

import re
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# 1. Configuration
DATA_FILE   = r"C:\Users\LENOVO\Desktop\COS30049-Computing Technology Innovation Project\Assignment 2 AI-based Project for cybersecurity (AI4Cyber)\Spam-50k.csv"
N_CLUSTERS  = 4
TOP_WORDS   = 10
RANDOM_SEED = 42
OUTPUT_IMG  = "spam_clusters_visualization.png"

CLUSTER_LABELS = {
    0: "General Spam / Phishing",               # free, click, new, message, time, info, company
    1: "Software Piracy / Cracked Programs",    # adobe, software, windows, professional, office, microsoft, pro, photoshop
    2: "Pharmacy / Drug Spam",                  # online, viagra, prescription, cialis, best, drugs, order, meds, pills
    3: "B2B / Branding / Marketing Services",   # business, logo, stationery, offers, success, identity, visual
}

stop_words = set(stopwords.words("english"))
noise_tokens = {"number_placeholder", "url_placeholder", "email_placeholder", 
                "http", "https", "com", "www", "html", "email", "save", "andmanyother", "nice", "hello"}
stop_words.update(noise_tokens)

# 2. Text cleaning function
def clean_text(text: str) -> str:
    """Normalize text: lowercase, replace URLs/emails/numbers, remove noise."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "url_placeholder", text)
    text = re.sub(r"\S+@\S+",                 "email_placeholder", text)
    text = re.sub(r"\d+",                     "number_placeholder", text)
    text = re.sub(r'[^\w\s!$?\-€£]', '',     text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


# 1. Load data
print("=" * 60)
print("SPAM TOPIC CLUSTERING  (K-Means Unsupervised)")
print("=" * 60)

print(f"\n1. Loading '{DATA_FILE}'...")
df = pd.read_csv(DATA_FILE, usecols=["Subject", "Message", "Spam/Ham"],
                 low_memory=False)
df["Spam/Ham"] = df["Spam/Ham"].astype(str).str.lower().str.strip()

# Keep only spam rows with non-empty messages
df_spam = df[df["Spam/Ham"] == "spam"].copy()
df_spam["Subject"] = df_spam["Subject"].fillna("")
df_spam["Message"]  = df_spam["Message"].fillna("").str[:600]
df_spam["text"]     = df_spam["Subject"] + " " + df_spam["Message"]
df_spam = df_spam[df_spam["text"].str.strip() != ""].reset_index(drop=True)

print(f"   Spam rows loaded: {len(df_spam):,}")


# 2. Clean text
print("\n2. Cleaning text...")
df_spam["cleaned"] = df_spam["text"].apply(clean_text)
df_spam = df_spam[df_spam["cleaned"].str.strip() != ""].reset_index(drop=True)
print(f"   Rows after cleaning: {len(df_spam):,}")


# 3. Vectorize with TF-IDF
print("\n3. Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    stop_words="english",
)
X = tfidf.fit_transform(df_spam["cleaned"])
print(f"   TF-IDF matrix shape: {X.shape}")


# 4. K-Means Clustering
print(f"\n4. Training K-Means with k={N_CLUSTERS}...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10,
            max_iter=300)
km.fit(X)
df_spam["cluster"] = km.labels_

# Evaluate clustering quality with silhouette score (range: -1 to 1)
sil_score = silhouette_score(X, km.labels_)

print(f"   Inertia: {km.inertia_:.2f}")
print(f"   Silhouette Score: {sil_score:.4f}")
print(f"   Cluster sizes: {dict(pd.Series(km.labels_).value_counts().sort_index())}")


# 5. Top keywords per cluster
terms = tfidf.get_feature_names_out()
centroids = km.cluster_centers_ 

print("\n" + "=" * 60)
print("TOP KEYWORDS PER CLUSTER")
print("=" * 60)

for i in range(N_CLUSTERS):
    top_idx  = centroids[i].argsort()[::-1][:TOP_WORDS]
    top_kws  = [terms[j] for j in top_idx]
    label    = CLUSTER_LABELS.get(i, f"Cluster {i}")
    print(f"\n  Cluster {i} - {label}:")
    print(f"    {', '.join(top_kws)}")

print("\n  → Justification:")
print("    K-Means clusters reveal distinct spam campaign themes - e.g.")
print("    financial lures, credential-phishing messages, promotional spam,")
print("    and malware-delivery emails - enabling early detection of NEW")
print("    campaigns by pattern-matching against known cluster centroids.")


# 6. PCA Visualization
print(f"\n5. Generating PCA scatter plot → '{OUTPUT_IMG}'...")

# Reduce to 2-D for visualization
X_norm = normalize(X, norm="l2")
pca = PCA(n_components=2, random_state=RANDOM_SEED)
coords = pca.fit_transform(X_norm.toarray())

fig, ax = plt.subplots(figsize=(10, 7))
colors = cm.tab10(np.linspace(0, 0.4, N_CLUSTERS))

for i in range(N_CLUSTERS):
    mask = df_spam["cluster"] == i
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=[colors[i]], label=f"Cluster {i}: {CLUSTER_LABELS.get(i, '')}",
        alpha=0.55, s=18, edgecolors="none",
    )

# Annotate cluster centers
for i in range(N_CLUSTERS):
    mask = df_spam["cluster"] == i
    cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
    ax.scatter(cx, cy, c="black", marker="X", s=150, zorder=5)
    ax.annotate(f"C{i}", (cx, cy), textcoords="offset points",
                xytext=(6, 6), fontsize=9, fontweight="bold")

ax.set_title("K-Means Spam Topic Clustering (PCA Projection)",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches="tight")
plt.close()

print(f"   Saved → {OUTPUT_IMG}")

print("\n" + "=" * 60)
print("CLUSTERING COMPLETE")
print("=" * 60)
print(f"  Clusters  : {N_CLUSTERS}")
print(f"  Spam rows : {len(df_spam):,}")
print(f"  Plot      : {OUTPUT_IMG}")
