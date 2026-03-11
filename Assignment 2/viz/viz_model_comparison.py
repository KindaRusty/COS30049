"""
viz_model_comparison.py
=======================
Visualizations to compare metrics across multiple models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_metrics_comparison(results: dict, return_fig=False):
    """Plot grouped bar chart for Accuracy, Precision, Recall, and F1-Score."""
    df_metrics = pd.DataFrame({
        name: {k: v for k, v in res.items() if k in ["Accuracy", "Precision", "Recall", "F1-Score"]}
        for name, res in results.items()
    }).T
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df_metrics.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
    
    ax.set_title("Model Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.xticks(rotation=15, ha="right")
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
        
    plt.tight_layout()
    if return_fig: return fig
    plt.show()

def plot_roc_curves(models: dict, X_test, y_test, return_fig=False):
    """Plot ROC curves for models that support predict_proba or decision_function."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']
    
    for i, (name, model) in enumerate(models.items()):
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                continue
                
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
        except Exception as e:
            print(f"Could not plot ROC for {name}: {e}")
            
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if return_fig: return fig
    plt.show()

if __name__ == "__main__":
    print("Model comparison visualization module ready.")
