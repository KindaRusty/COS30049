# TechNova - Spam Detection (AI-based Project for Cybersecurity)

**Project:** COS30049 - Computing Technology Innovation Project (Assignment 2)  
**Task:** Build an AI-based system to detect Spam/Phishing messages.

---

## Team & Project Overview
This project is a complete Machine Learning Pipeline designed to classify and detect spam messages/emails. It fulfills all the advanced criteria required in Assignment 2.

**Key Features:**
1. **Complete Pipeline (10 pts):** Encompasses Text Processing (NLTK), TF-IDF Vectorization, and the evaluation of four Machine Learning algorithms (Naive Bayes, SVM, Logistic Regression, Random Forest).
2. **Datasets (4 pts):** Utilizes `Spam-50k.csv` as the primary dataset.
3. **Additional Dataset Generation (4 pts):** Employs the NLP Augmentation library (`nlpaug`) to generate a synthetic dataset (`synthetic_nlpaug.csv`). This simulates spam variations to **stress-test** the model's robustness.
4. **Additional ML Algorithms (4 pts):** 
   - **K-Means Clustering:** Unsupervised clustering to discover latent topics within Spam campaigns.
   - **Metadata Pipeline:** A Scikit-Learn Pipeline combining text content with numerical features (length, special characters, time) using `ColumnTransformer`.
5. **Clean Code Structure (2 pts):** The source code is intelligently divided into an **interactive Notebook** for easy grading and **pure modular code files (.py)** for logical separation.

---

## Directory Structure and File Descriptions

The project is structured into a main execution file (`.ipynb`) and source code directories (`src/` and `viz/`). Here is the function of each file:

```text
├── SpamDetection_AI4Cyber.ipynb      # MAIN FILE (Notebook): Contains the entire, fully executable pipeline. 
├── README.md                         # This instruction file
├── Spam-50k.csv                      # Required source dataset to run the code
│
├── src/                              # Contains Pure Code Files (Logic only, no plotting commands)
│   ├── 01_data_preprocessing.py      # Cleans data, removes Urls/Emails, tokenizes text, extracts metadata
│   ├── 02_model_training.py          # Trains 4 baseline Machine Learning models (NB, SVM, LR, RF)
│   ├── 03_hyperparameter_tuning.py   # GridSearchCV implementation using PredefinedSplit (prevents Data Leakage)
│   ├── 04_metadata_pipeline.py       # Pipeline combining Metadata and Text features (TF-IDF + MinMaxScaler)
│   └── 05_clustering_augmentation.py # K-Means Clustering algorithm and synthetic data generation via nlpaug
│
└── viz/                              # Contains Visualization Code Files
    ├── viz_eda.py                    # Plots EDA charts (Pie chart, Bar chart class, Text length boxes)
    ├── viz_model_comparison.py       # Plots Acc/Prec/Recall/F1 comparison charts between models
    ├── viz_confusion_matrices.py     # Generates Model Confusion Matrix Heatmap images
    ├── viz_cross_validation.py       # Plots bar charts of scores after 5-Fold Cross-Validation
    └── viz_clustering.py             # Plots 2D Scatter graphs for K-Means Spam Clustering results
```

---

## Environment Setup Guide

Before running any file, the required libraries must be installed.
Please open CMD / Terminal (in the directory containing the files) and run the following commands:

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install necessary Data Science & Machine Learning libraries
pip install pandas numpy scikit-learn matplotlib seaborn joblib nlpaug
```

*(Note: To run the Notebook, you must have Jupyter Notebook or a VS Code environment. It can be installed via `pip install notebook`)*

---

## Execution Guide (For Grading)

### Method 1: Run the Jupyter Notebook File (Recommended - Easiest to read and grade)

This file aggregates the entire assignment with detailed explanations for each part. It automatically outputs charts and reports with just a "Run All" action.

1. **Open the file** `SpamDetection_AI4Cyber.ipynb` (using VS Code, Jupyter Notebook, Google Colab...).
2. Ensure the `Spam-50k.csv` file is located directly in the same directory as the Notebook.
3. Click the **Run All** button (or run each cell `Shift + Enter` from top to bottom).
4. *Note: At **Section 7: Data Augmentation**, the algorithm will automatically generate the `synthetic_nlpaug.csv` file for the Stress-Test. This generation process may take 1 to 3 minutes depending on machine speed.*
5. Once the Notebook finishes running, 2 trained model files will be generated in a subdirectory:
   - `saved_models/spam_pipeline.pkl` (The actual model used for future applications)
   - `saved_models/label_encoder.pkl` (The Ham/Spam label converter)
   - *These are prepared for Assignment 3 GUI/Web integration.*

---

### Method 2: Run Individual Code Module Scripts (CLI Execution)

Besides the Notebook, we've modularized the Machine Learning problem into several separate files to ensure clean code (as described in the file structure). You can run and independently test each feature cluster from the Command Line:

> **Note: Set your Working Directory to the location of this README file.**

**1. Test the Data Cleaning Script (Data Preprocessing)**
Run this to see how the Pipeline removes redundant characters, malicious links/urls, etc.
```bash
python src/01_data_preprocessing.py
```

**2. Plot General EDA Charts (Exploratory Data Analysis)**
Run this visualization file to view Pie Charts and Histograms of the Dataset.
```bash
python viz/viz_eda.py
```

**3. Test the Main Algorithm Training Script**
```bash
python src/02_model_training.py
```

**4. Run Additional Algorithms - Generate Synthetic Data and K-Means Clustering**
```bash
python src/05_clustering_augmentation.py
```

All files include an `if __name__ == "__main__":` block to ensure they run cleanly and independently (Modularization).

---

## Quick Model Testing Guide 
*(Corresponding to Core Features execution -> Saved Models Predict describe)*

A convenient script at the end of the Notebook is provided to test the model's prediction capabilities as follows:

```python
import joblib

# Can only be run after the Notebook has finished training and created the .pkl files
pipeline = joblib.load('saved_models/spam_pipeline.pkl')
encoder = joblib.load('saved_models/label_encoder.pkl')

# Test Data
raw_text = "URGENT! You have won a $1,000 Walmart Gift Card. Click here to claim your prize instantly!"
pred_idx = pipeline.predict([raw_text])[0]
label = encoder.inverse_transform([pred_idx])[0]

print(f"Prediction: {label.upper()}")
# Expected Output: SPAM
```

---
*Developed for TechNova Computing Technology Innovation Project - Assignment 2.*
