"""
viz_eda.py
==========
Visualization tools for Exploratory Data Analysis (EDA).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def plot_class_distribution(df: pd.DataFrame, label_col: str = "Spam/Ham", return_fig=False):
    """Plot distribution of Spam vs Ham."""
    counts = df[label_col].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart visualization for class balance
    ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'], explode=(0, 0.1))
    ax1.set_title("Class Distribution (Pie)", fontweight='bold')
    
    # Bar chart for exact volume comparison
    sns.barplot(x=counts.index, y=counts.values, ax=ax2, palette=['#3498db', '#e74c3c'])
    ax2.set_title("Class Distribution (Bar)", fontweight='bold')
    ax2.set_ylabel("Count")
    
    for i, v in enumerate(counts.values):
        ax2.text(i, v + (v*0.01), str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()

def plot_text_length_distribution(df: pd.DataFrame, text_col: str = "cleaned_text", label_col: str = "Spam/Ham", return_fig=False):
    """Plot the distribution of text lengths."""
    df['temp_length'] = df[text_col].apply(lambda x: len(str(x)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='temp_length', hue=label_col, bins=50, kde=True, ax=ax, palette=['#3498db', '#e74c3c'])
    
    ax.set_title("Distribution of Text Lengths by Class", fontsize=14, fontweight='bold')
    ax.set_xlabel("Text Length (characters)")
    # Clip extreme outlines for better visual
    if df['temp_length'].max() > 5000:
        ax.set_xlim(0, 5000)
    
    plt.tight_layout()
    df.drop(columns=['temp_length'], inplace=True)
    if return_fig: return fig
    plt.show()

def plot_word_count_boxplot(df: pd.DataFrame, text_col: str = "cleaned_text", label_col: str = "Spam/Ham", return_fig=False):
    """Boxplot of word counts between Spam and Ham."""
    df['temp_words'] = df[text_col].apply(lambda x: len(str(x).split()))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=label_col, y='temp_words', data=df, ax=ax, palette=['#3498db', '#e74c3c'], showfliers=False)
    
    ax.set_title("Word Count Boxplot (Outliers Hidden)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Word Count")
    
    plt.tight_layout()
    df.drop(columns=['temp_words'], inplace=True)
    if return_fig: return fig
    plt.show()

if __name__ == "__main__":
    print("EDA visualization module ready.")
