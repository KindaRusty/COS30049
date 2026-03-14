"""
viz_cross_validation.py
=======================
Visualizes the results of k-fold cross-validation.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_cv_results(cv_results: dict, return_fig=False):
    """Plot bar chart showing the Mean F1-Score and standard error for models during CV."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(cv_results.keys())
    means = [cv_results[n]["Mean F1"] for n in names]
    stds = [cv_results[n]["Std F1"] for n in names]
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    
    # Display bars with error bars representing Standard Deviation (yerr)
    # This visualizes the stability and consistency of the model scores across folds
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors[:len(names)], edgecolor="black")
    
    ax.set_title("5-Fold Cross-Validation — Mean F1-Score", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1-Score")
    # Adjust ylim dynamically based on means to zoom in on differences
    min_mean = min(means) if means else 0.7
    ax.set_ylim(max(0.0, min_mean - 0.1), 1.05)
    ax.tick_params(axis="x", rotation=10)
    
    # Annotate bars with precise mean scores
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.4f}", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    if return_fig: return fig
    plt.show()

if __name__ == "__main__":
    print("Cross validation visualization module ready.")
