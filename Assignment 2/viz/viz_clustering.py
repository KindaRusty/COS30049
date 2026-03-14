"""
viz_clustering.py
=================
Visualizations for K-Means topic clustering via PCA projection and top keywords.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

CLUSTER_LABELS = {
    0: "Phishing / General Spam",               
    1: "Software Piracy / Cracked Programs",    
    2: "Pharmacy / Drug Spam",                  
    3: "B2B / Branding / Marketing"             
}

def plot_pca_scatter(df_spam: pd.DataFrame, tfidf_matrix, cluster_col: str = "cluster", n_clusters=4, random_seed=42, return_fig=False):
    """Generate PCA scatter plot of spam clusters."""
    # Reduce to 2-D for visualization
    X_norm = normalize(tfidf_matrix, norm="l2") # L2 normalize to prevent magnitude-dominant PCA results
    pca = PCA(n_components=2, random_state=random_seed) # Dimensionality reduction for 2D visualization
    coords = pca.fit_transform(X_norm.toarray())
    
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = cm.tab10(np.linspace(0, 0.4, n_clusters))
    
    for i in range(n_clusters):
        mask = df_spam[cluster_col] == i
        if not any(mask): continue
        # Plot each cluster with a unique color from the tab10 palette
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=f"Cluster {i}: {CLUSTER_LABELS.get(i, '')}",
            alpha=0.6, s=25, edgecolors="white", linewidth=0.5
        )
        
    # Annotate cluster centers (computed as mean position of members in 2D space) for clarity
    for i in range(n_clusters):
        mask = df_spam[cluster_col] == i
        if not any(mask): continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        # Mark the centroid with a bold Black 'X'
        ax.scatter(cx, cy, c="black", marker="X", s=200, zorder=5, edgecolors="white", linewidth=1.5)
        # Add label for the centroid
        ax.annotate(f"C{i}", (cx, cy), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

    ax.set_title("K-Means Spam Topic Clustering (PCA Projection)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    
    # Adjust legend position to fit better
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9, title="Identified Campaigns", title_fontsize="11")
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    if return_fig: return fig
    plt.show()

def plot_top_keywords_per_cluster(top_keywords: dict, return_fig=False):
    """Plot a horizontal bar chart showing top keywords for each cluster."""
    n_clusters = len(top_keywords)
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 6), sharex=True)
    
    if n_clusters == 1:
        axes = [axes]
        
    colors = cm.tab10(np.linspace(0, 0.4, n_clusters))
    
    for i, (cluster_name, keywords) in enumerate(top_keywords.items()):
        ax = axes[i]
        
        # Assign descending weights because centroid features are sorted by relative importance
        n = len(keywords)
        weights = range(n, 0, -1) 
        
        y_pos = np.arange(len(keywords))
        
        ax.barh(y_pos, weights, align='center', color=colors[i], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keywords)
        ax.invert_yaxis()  # labels read top-to-bottom
        
        label = CLUSTER_LABELS.get(i, "")
        ax.set_title(f"{cluster_name}\n({label})", fontsize=11, fontweight="bold")
        ax.set_xlabel('Relative Importance')
        
    plt.suptitle('Top Keywords Driving Each Spam Campaign', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    if return_fig: return fig
    plt.show()

if __name__ == "__main__":
    print("Clustering visualization module ready.")
