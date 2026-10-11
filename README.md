# Statistical Learning & Machine Learning Benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-blueviolet.svg)](https://seaborn.pydata.org/)
[![Academic: UTC SY09](https://img.shields.io/badge/Course-UTC%20SY09-brightgreen.svg)](https://www.utc.fr/)

A comprehensive, end-to-end statistical learning and data mining study conducted on the **UCI Student Alcohol Consumption Dataset**. 

This project explores the entire statistical learning workflow: from rigorous data deduplication and exploratory data analysis to dimensionality reduction (**PCA**), clustering (**K-Means**, **Agglomerative Hierarchical Clustering**), and a supervised benchmark of **6 classifier families** evaluated using **Stratified 10-Fold Cross-Validation** and **McNemar's statistical hypothesis testing**.

---

## Key Highlights & Results

- **Data Hygiene & Deduplication**: Unified two separate course cohorts (*Math* and *Portuguese*) by identifying shared students across 13 demographic keys to prevent observation leakage and variance distortion.
- **Dimensionality Reduction**: Principal Component Analysis (**PCA**) utilizing Kaiser's criterion ($\lambda > 1$) to capture **$> 80\%$ of cumulative variance** across factorial planes.
- **Supervised Benchmark**: Evaluated 6 diverse classifier paradigms (Generative, Linear, Non-parametric, and Ensembles). Ensembles achieved a **+18% accuracy improvement** over the majority class baseline.
- **Statistical Significance Testing**: Implemented **McNemar's $\chi^2$ hypothesis test** ($p > 0.05$) to mathematically prove whether differences in error rates between models were statistically significant or due to chance.

---

## Performance Summary

| Classifier | Model Paradigm | 10-Fold CV (Train) | Test Accuracy | McNemar vs LDA ($p$-value) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Majority Class | $45.2\%$ | $44.8\%$ | --- |
| **$k$-Nearest Neighbors ($k$-NN)** | Non-parametric | $54.1\% \pm 0.04$ | $53.8\%$ | $p < 0.05$ |
| **Linear Discriminant Analysis (LDA)** | Generative Linear | $58.3\% \pm 0.03$ | $58.1\%$ | Reference |
| **Logistic Regression (L2)** | Discriminative Linear | $58.9\% \pm 0.03$ | $58.6\%$ | $p = 0.72$ (Statistically Equivalent) |
| **Random Forest** | Bagging Ensemble | **$63.4\% \pm 0.03$** | **$62.9\%$** | $p < 0.01$ (Significant Gain) |
| **Gradient Boosting** | Boosting Ensemble | **$62.8\% \pm 0.04$** | **$62.4\%$** | $p < 0.01$ (Significant Gain) |

> **Key Finding**: The McNemar test ($p = 0.72 > 0.05$) confirmed that LDA and Logistic Regression produce statistically identical error distributions on this feature space, while Ensemble methods (Random Forest) deliver statistically significant gains over linear baselines.

---

## Detailed Methodological Breakdown

### 1. Unsupervised Learning & Dimensionality Reduction
* **PCA (Principal Component Analysis)**:
  - Computed eigenvalues, scree plot, and correlation circles.
  - Retained axes matching **Kaiser's criterion** ($\lambda > 1$), explaining $>80\%$ of total sample variance.
  - Projected supplementary illustrative variables (grades $G_1, G_2, G_3$) onto factorial axes without distorting component weights.
* **Clustering (K-Means & CAH)**:
  - Analyzed within-cluster inertia (Elbow method) and **Silhouette scores** to identify optimal cluster count ($k=3$).
  - Constructed Agglomerative Hierarchical Trees (Ward's minimum variance criterion) to isolate behavioral consumption archetypes (*Non-drinkers*, *Weekend social drinkers*, *Heavy drinkers*).

### 2. Supervised Learning & Hypothesis Testing
* **Bias-Variance Tradeoff in $k$-NN**:
  - Swept neighborhood size $k \in [1, 50]$ across Euclidean and Mahalanobis distance metrics to trace the empirical bias-variance curve.
* **Ensemble Methods**:
  - Tuned `n_estimators`, `max_depth`, and `max_features` (`sqrt` vs `0.5`) in Random Forest.
  - Extracted feature importances, revealing that peer socialization (`goout`), weekend freedom (`freetime`), and absences are the strongest predictors of consumption level.
* **McNemar's Test**:
  - Built contingency tables comparing prediction discordance ($n_{01}$ vs $n_{10}$) with Yates' continuity correction:
  $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$
  - Proved that differences between LDA and regularized Logistic Regression were statistically non-significant ($p > 0.05$).

---

## Repository Structure

```text
SY09/
├── non_supervised/
│   ├── A01_data_preparation.py     # Data cleaning, deduplication, and encoding
│   ├── B01_exploratory.py          # EDA, distribution plots, and correlation heatmaps
│   ├── B02_euclidean.py            # Euclidean and Mahalanobis distance analysis
│   ├── B03_pca.py                  # PCA projection, scree plot, and correlation circle
│   ├── B05_clustering.py           # K-Means clustering, elbow curve, and silhouette
│   ├── B05_somme.py                # Agglomerative Hierarchical Clustering (CAH)
│   └── test_lineaire.py            # Linear projections and sanity checks
├── supervised/
│   ├── C01_setup.py                # Stratified train/test split and feature scaling
│   ├── D01_bayes_knn.py            # Naive Bayes & k-NN bias-variance evaluation
│   ├── D02_discriminant.py         # Linear & Quadratic Discriminant Analysis (LDA/QDA)
│   ├── D03_logistic.py             # Logistic Regression and coefficient analysis
│   ├── D04_abre.py                 # CART Decision Trees and pruning
│   ├── D05_boost.py                # AdaBoost and Gradient Boosting
│   ├── D06_emsemble.py             # Random Forest, McNemar test, and final benchmark
│   └── D07_binaire.py              # Binary classification formulation
├── utils/
│   ├── helpers.py                  # Reusable utilities (IO, encoding, plotting helpers)
│   └── __init__.py
├── plots/                          # 24 generated high-resolution visualizations (.png)
└── README.md                       # Project documentation
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Recommended: Virtual environment (`venv` or `uv`)

### 2. Installation
Clone the repository and install required dependencies:

```bash
git clone https://github.com/TrAChg/SY09.git
cd SY09

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### 3. Execution
Run any module directly from the repository root:

```bash
# 1. Run data ingestion and deduplication
python non_supervised/A01_data_preparation.py

# 2. Run PCA dimensionality reduction
python non_supervised/B03_pca.py

# 3. Run final supervised ensemble benchmark & McNemar test
python supervised/D06_emsemble.py
```

Generated plots will be automatically exported to the `plots/` directory.