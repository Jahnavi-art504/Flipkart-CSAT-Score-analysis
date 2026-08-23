# 📊 Flipkart Customer Satisfaction (CSAT) Score Analysis & Prediction

> **End-to-end data science project** — Exploratory Data Analysis + Machine Learning classification on 85,907 Flipkart customer support interactions to understand and predict customer satisfaction scores.

---

## 📌 Project Overview

Customer satisfaction is a critical metric for any e-commerce platform. This project analyzes Flipkart's customer support data to:

- **Uncover patterns** in customer complaints, agent performance, and response times
- **Identify key drivers** of customer satisfaction and dissatisfaction
- **Build and evaluate ML models** to predict CSAT score categories (Very Poor → Excellent)

This repository contains two Jupyter notebooks:

| Notebook | Description |
|---|---|
| `Flipkart_CSAT_Score_EDA_by_SM_.ipynb` | Exploratory Data Analysis — 20+ visualizations following the UBM framework |
| `Flipkart_CAST_Score_ML_by__SM.ipynb` | Machine Learning — Feature engineering, NLP, SMOTE, 3 classification models with hyperparameter tuning |

---

## 📂 Dataset

- **Source:** Flipkart Customer Support Data
- **Records:** 85,907 rows × 19 columns
- **Target Variable:** `CSAT Score` (1–5 scale → classified as Very Poor / Poor / Average / Good / Excellent)

### Key Features

| Feature | Description |
|---|---|
| `channel_name` | Support channel (Inbound, Outcall, Email) |
| `category` / `Sub-category` | Issue type and sub-type |
| `Customer Remarks` | Free-text customer feedback |
| `Product_category` | Product type (Electronics, Fashion, etc.) |
| `Item_price` | Price of the purchased item |
| `connected_handling_time` | Call/chat handling duration (seconds) |
| `Agent Shift` | Agent working shift (Morning / Evening / Night) |
| `Tenure Bucket` | Agent experience bucket (0–30, 31–90, >90 days) |
| `Time Difference` | *(Engineered)* Issue response time in minutes |

---

## 🔍 Part 1 — Exploratory Data Analysis

### Data Cleaning & Wrangling
- Converted timestamp columns to `datetime` format
- Engineered `Time Difference` feature (response time in minutes)
- Clipped outliers in `Item_price` and `Time Difference` at the 99th percentile
- Filled `Customer_City` missing values with `'Unknown'`

### Visualizations (UBM Framework)
20+ charts covering:

- **Univariate:** CSAT score distribution, channel distribution, item price histograms
- **Bivariate:** Response time vs. agent shift, CSAT by product category, CSAT by channel, average CSAT by agent tenure
- **Multivariate:** Correlation heatmaps, pair plots, CSAT by manager × shift

### Key EDA Insights
- 📈 Most CSAT scores are 4–5, but a significant segment of 1–2 scores signals service gaps
- 🌙 Evening shifts have longer response times and lower average CSAT scores
- 📱 Electronics category consistently shows lower satisfaction compared to other categories
- ⏱️ Response time negatively correlates with CSAT — faster responses lead to higher scores

### Hypothesis Testing

| Hypothesis | Test Used | Conclusion |
|---|---|---|
| Response time affects CSAT | Spearman's Rank Correlation | Significant relationship found |
| Agent shift affects CSAT | Kruskal-Wallis H Test | Significant difference across shifts |
| Item price affects CSAT | Statistical correlation test | Tested for significance |

---

## 🤖 Part 2 — Machine Learning

### Feature Engineering & NLP
- **TF-IDF Vectorization** on `Customer Remarks` (100 features)
- **PCA** applied to TF-IDF features → reduced to 10 principal components
- **Label Encoding** for categorical variables
- **StandardScaler** for numerical feature normalization
- **SMOTE** (Synthetic Minority Oversampling Technique) to handle class imbalance in training data

### Models Implemented

| Model | Tuning Method |
|---|---|
| Random Forest Classifier | RandomizedSearchCV |
| Logistic Regression | GridSearchCV |
| Decision Tree Classifier | GridSearchCV |

Each model is evaluated with:
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score per class)
- Confusion Matrix
- Cross-Validation scores
- Pre vs. Post hyperparameter tuning comparison

---

## 🛠️ Tech Stack

```
Python 3.x
pandas · numpy · matplotlib · seaborn
scikit-learn · imbalanced-learn (SMOTE)
nltk · contractions (NLP preprocessing)
scipy (hypothesis testing)
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SumitMaske8055/flipkart-csat-analysis.git
cd flipkart-csat-analysis
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn nltk scipy contractions
```

### 3. Download NLTK resources
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### 4. Add the dataset
Place `Customer_support_data.csv` in the project root directory.

### 5. Run the notebooks
Open and run in order:
1. `Flipkart_CSAT_Score_EDA_by_SM_.ipynb`
2. `Flipkart_CAST_Score_ML_by__SM.ipynb`

> ✅ Both notebooks are designed to run end-to-end without errors (Production Grade Code).

---

## 📁 Repository Structure

```
flipkart-csat-analysis/
│
├── Flipkart_CSAT_Score_EDA_by_SM_.ipynb   # EDA notebook
├── Flipkart_CAST_Score_ML_by__SM.ipynb    # ML notebook
├── Customer_support_data.csv              # Dataset (add manually)
└── README.md
```

---

## 💡 Business Impact

| Insight | Recommended Action |
|---|---|
| Evening shift has lower CSAT | Improve staffing or training during evening hours |
| Electronics category has low satisfaction | Specialized training for electronics support agents |
| Long response times hurt CSAT | Set SLA targets and monitor response time KPIs |
| Low CSAT scores cluster around unresolved issues | Build escalation workflows for repeat complaints |
