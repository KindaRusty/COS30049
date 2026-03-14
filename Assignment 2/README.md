# TechNova - Spam Detection (AI-based Project for Cybersecurity)

**Project:** COS30049 - Computing Technology Innovation Project (Assignment 2)  
**Task:** Build an AI-based system to detect Spam/Phishing messages.

---

## Team & Project Overview
This project is a complete Machine Learning Pipeline designed to classify and detect spam messages/emails. It fulfills all the advanced criteria required in Assignment 2.

**Key Features:**
1. **Complete Pipeline:** Encompasses Text Processing (NLTK), TF-IDF Vectorization, and the evaluation of four Machine Learning algorithms (Naive Bayes, SVM, Logistic Regression, Random Forest).
2. **Datasets:** Utilizes `Spam-50k.csv` as the primary dataset, merged with `spam_Emails_data.csv` (~220k emails) for a combined ~194k unique training rows.
3. **Additional Dataset Generation:** Employs the NLP Augmentation library (`nlpaug`) to generate a synthetic dataset (`synthetic_nlpaug.csv`). This simulates spam variations to **stress-test** the model's robustness.
4. **Additional ML Algorithms:** 
   - **K-Means Clustering:** Unsupervised clustering to discover latent topics within Spam campaigns.
   - **Metadata Pipeline:** A Scikit-Learn Pipeline combining text content with numerical features (length, special characters, time) using `ColumnTransformer`.
5. **Clean Code Structure:** The source code is intelligently divided into an **interactive Notebook** for easy grading and **pure modular code files (.py)** for logical separation.

---

## Directory Structure and File Descriptions

```text
├── SpamDetection_AI4Cyber.ipynb      # MAIN FILE 
├── README.md                         # This instruction file
├── training_dataset_final.csv        # PROCESSED dataset used by the final model
├── Spam-50k.csv                      # Raw source dataset 1 (required to regenerate training data)
├── synthetic_nlpaug.csv              # Augmented dataset for robustness stress-test
│
├── src/                              # Contains Pure Code Files (Logic only, no plotting commands)
│   ├── data_preprocessing.py         # Cleans data, removes URLs/Emails, tokenizes text, extracts metadata
│   ├── model_training.py             # Trains 4 baseline Machine Learning models (NB, SVM, LR, RF)
│   ├── hyperparameter_tuning.py      # GridSearchCV implementation using PredefinedSplit (prevents Data Leakage)
│   ├── metadata_pipeline.py          # Pipeline combining Metadata and Text features (TF-IDF + MinMaxScaler)
│   └── clustering_augmentation.py    # K-Means Clustering algorithm and synthetic data generation via nlpaug
│
├── viz/                              # Contains Visualization Code Files
│   ├── viz_eda.py                    # Plots EDA charts (Pie chart, Bar chart class, Text length boxes)
│   ├── viz_model_comparison.py       # Plots Acc/Prec/Recall/F1 comparison charts between models
│   ├── viz_confusion_matrices.py     # Generates Model Confusion Matrix Heatmap images
│   ├── viz_cross_validation.py       # Plots bar charts of scores after 5-Fold Cross-Validation
│   └── viz_clustering.py             # Plots 2D Scatter graphs for K-Means Spam Clustering results
│
├── saved_models/                     # Auto-generated after training
│   ├── spam_pipeline.pkl             # Trained model pipeline (TF-IDF + Classifier)
│   └── label_encoder.pkl             # Label encoder (Ham/Spam ↔ 0/1)
│
└── export_training_data.py           # Utility script to regenerate training_dataset_final.csv
```

---

## 1. Environment Configuration (Conda)

Set up a reproducible Python environment using **conda**.  
Open a terminal (Anaconda Prompt, CMD, or PowerShell) and run:

```bash
# 1. Create a new conda environment with Python 3.10
conda create -n ai4cyber python=3.10 -y

# 2. Activate the environment
conda activate ai4cyber

# 3. Install core Data Science & Machine Learning libraries via conda
conda install pandas numpy scikit-learn matplotlib seaborn joblib nltk -y

# 4. Install nlpaug via pip (not available on conda default channels)
pip install nlpaug

# 5. Install Jupyter to run the Notebook
conda install jupyter -y
```

## 2. Data Processing

### 2.1 About the Processed Dataset

The file `training_dataset_final.csv` is the **final processed dataset** used to train the model. It was made by:

1. Loading and merging `Spam-50k.csv` (primary) + `spam_Emails_data.csv` (additional ~220k rows)
2. Cleaning: standardizing labels, filling NaN values, combining Subject + Message
3. Text preprocessing: lowercasing, replacing URLs/emails/numbers with placeholders, removing stopwords via NLTK
4. Feature engineering: extracting `text_length`, `word_count`, `special_char_count`, `hour`, `is_weekend`
5. Label encoding: `ham → 0`, `spam → 1`
6. Deduplication: removing duplicate messages

**Final dataset columns:**

| Column | Description |
|--------|-------------|
| `Spam/Ham` | Original text label (`ham` or `spam`) |
| `cleaned_text` | Processed text after cleaning pipeline |
| `text_length` | Number of characters in the original combined text |
| `word_count` | Number of words in the original combined text |
| `special_char_count` | Count of punctuation/special characters |
| `hour` | Hour extracted from email timestamp (default 12 if missing) |
| `is_weekend` | 1 if email was sent on Saturday/Sunday, else 0 |
| `label` | Encoded label (0 = ham, 1 = spam) |

### 2.2 Regenerating the Processed Dataset (Optional)

If you want to regenerate `training_dataset_final.csv` from raw data, ensure both source CSV files are in the project root directory, then run:

```bash
conda activate ai4cyber
python export_training_data.py
```

This will run the full preprocessing pipeline and output `training_dataset_final.csv`.

### 2.3 Testing the Preprocessing Module Independently

```bash
python src/data_preprocessing.py
```

---

## 3. Model Training (Approximately takes 30 mins on i5-12450hx CPU with RTX3050 6gb GPU)

### 3.1 Method 1: Run the Jupyter Notebook (Recommended - Easiest to Grade)

This file aggregates the entire assignment with detailed explanations for each part.

1. **Open** `SpamDetection_AI4Cyber.ipynb` (Recommend using VS Code).
2. Ensure `Spam-50k.csv` and `spam_Emails_data.csv` are in the same directory as the Notebook.
3. Click **Run All**.
4. *Note: At **Section 7: Data Augmentation**, the algorithm generates `synthetic_nlpaug.csv` for stress-testing. This may take 1–3 minutes.*
5. After completion, trained models will be saved in `saved_models/`:
   - `spam_pipeline.pkl` - Main trained pipeline (TF-IDF + Classifier)
   - `label_encoder.pkl` - Label converter (Ham ↔ Spam)

### 3.2 Method 2: Run Individual Code Modules (CLI)

> **Set your working directory** to the location of this README file.

**Train baseline models (4 algorithms):**
```bash
python src/model_training.py
```

**Run hyperparameter tuning (GridSearchCV with PredefinedSplit):**
```bash
python src/hyperparameter_tuning.py
```

**Run additional algorithms - K-Means Clustering + Synthetic Data Generation:**
```bash
python src/clustering_augmentation.py
```

**Run metadata-enhanced pipeline (TF-IDF + numerical features):**
```bash
python src/metadata_pipeline.py
```

### 3.3 Training Parameters (Key Configurations)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| TF-IDF `max_features` | 10,000 | Limits vocabulary to top 10k terms by frequency |
| TF-IDF `ngram_range` | (1, 2) | Captures both unigrams and bigrams for context |
| TF-IDF `min_df` / `max_df` | 5 / 0.8 | Filters rare and overly common terms |
| Train/Val/Test split | 70% / 15% / 15% | Stratified split preserving class distribution |
| GridSearchCV `cv` | PredefinedSplit | Uses validation set directly - prevents data leakage |
| K-Means `n_clusters` | 4 | Four spam campaign topics identified |
| NLP Augmentation | synonym, delete, char_swap | Three augmentation strategies for robustness testing |

### 3.4 Visualization Commands

```bash
python viz/viz_eda.py                    # EDA charts (class distribution, text lengths)
python viz/viz_model_comparison.py       # Model metrics comparison bar charts
python viz/viz_confusion_matrices.py     # Confusion matrix heatmaps
python viz/viz_cross_validation.py       # 5-Fold Cross-Validation results
python viz/viz_clustering.py             # K-Means PCA scatter plots
```

---

## 4. Model Prediction

### 4.1 Using the Saved Pipeline

After the Notebook has finished training (or models are available in `saved_models/`), use the following to predict new emails:

```python
import joblib

# Load the trained pipeline and label encoder
pipeline = joblib.load('saved_models/spam_pipeline.pkl')
encoder = joblib.load('saved_models/label_encoder.pkl')

# Test with a sample email
raw_text = "URGENT! You have won a $1,000 Walmart Gift Card. Click here to claim your prize instantly!"
pred_idx = pipeline.predict([raw_text])[0]
label = encoder.inverse_transform([pred_idx])[0]

print(f"Prediction: {label.upper()}")
# Expected Output: SPAM
```

### 4.2 Prediction Workflow Explained

```text
Raw Email Text
    ↓
TF-IDF Vectorization (built into the pipeline)
    ↓
Classifier Prediction (Logistic Regression / SVM / NB / RF)
    ↓
Label Decoding (0 → ham, 1 → spam)
    ↓
Final Output: "HAM" or "SPAM"
```

> **Note:** The saved `spam_pipeline.pkl` includes the TF-IDF vectorizer as part of the Scikit-Learn Pipeline, so you only need to pass raw text - no manual preprocessing required.

### 4.3 Batch Prediction Example

```python
import joblib

pipeline = joblib.load('saved_models/spam_pipeline.pkl')
encoder = joblib.load('saved_models/label_encoder.pkl')

emails = [
    "Hi John, let's meet at 3pm for the project discussion.",
    "FREE!!! Claim your lottery prize NOW! Click http://scam.link",
    "Please review the attached quarterly report before Friday.",
    "You've been selected for a $500 Amazon gift card! Act fast!"
]

for email in emails:
    pred = pipeline.predict([email])[0]
    label = encoder.inverse_transform([pred])[0]
    print(f"[{label.upper():4s}] {email[:60]}...")
```

---
*Developed for TechNova Computing Technology Innovation Project - Assignment 2.*
