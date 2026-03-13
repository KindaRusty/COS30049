"""
viz_confusion_matrices.py
=========================
Generates visual heatmaps of confusion matrices for evaluating True/False positives.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import math

def plot_all_confusion_matrices(results: dict, y_test, le, return_fig=False):
    """Plot confusion matrices for all evaluated models in a grid."""
    n_models = len(results)
    cols = 2
    rows = math.ceil(n_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    # Ensure axes is always a 2D array for consistent iteration logic
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx // cols, idx % cols] if rows > 1 else axes[0, idx % cols]
        
        y_pred = res["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        
        # Heatmap shows exact counts with integer formatting (fmt="d")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    annot_kws={"size": 14})
        
        ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")
        
    # Hide any unused subplots
    for idx in range(n_models, rows * cols):
        ax = axes[idx // cols, idx % cols] if rows > 1 else axes[0, idx % cols]
        ax.axis('off')

    plt.tight_layout()
    if return_fig: return fig
    plt.show()

def plot_single_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", cmap="Reds", return_fig=False):
    """Plot a single confusion matrix (e.g., for robustness test)."""
    # Create a single figure and subplot
    fig, ax = plt.subplots(figsize=(6, 5))
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    # Set title and labels for the plot
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    
    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    # Optionally return the figure object
    if return_fig: return fig
    # Display the plot
    plt.show()

if __name__ == "__main__":
    print("Confusion matrices visualization module ready.")
