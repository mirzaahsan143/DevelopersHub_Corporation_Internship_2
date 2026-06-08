# Loan Default Risk with Business Cost Optimization

**DevelopersHub Corporation – Data Science & Analytics Advanced Internship | Task 4**

---

## Overview

This project builds a machine learning pipeline to **predict loan defaults** and determine the **optimal decision threshold** that minimises total business cost. The core insight is that a naive 0.5 probability cutoff is rarely optimal — approving a defaulter and rejecting a good customer carry very different financial consequences that must be explicitly modelled.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Building](#5-model-building)
6. [Feature Importance](#6-feature-importance)
7. [Business Cost Optimization](#7-business-cost-optimization)
8. [Results & Conclusions](#8-results--conclusions)
9. [Generated Plots](#9-generated-plots)
10. [How to Run](#10-how-to-run)
11. [Dependencies](#11-dependencies)

---

## 1. Problem Statement

Banks lose money in two ways when making lending decisions:

| Error Type | Description | Business Cost |
|---|---|---|
| **False Negative (FN)** | Approving a loan that defaults | **$5,000** per case |
| **False Positive (FP)** | Rejecting a creditworthy applicant | **$1,200** per case (lost revenue) |
| True Negative | Correctly approving a good loan | $0 |
| True Positive | Correctly flagging a defaulter | $0 |

False Negatives are ~4× more expensive than False Positives, which means the model should lean conservative — i.e., use a **lower probability threshold** than the default 0.5.

---

## 2. Dataset

The notebook simulates a realistic version of the **Home Credit Default Risk** dataset (10,000 applicants, seed = 42). The target variable is `TARGET` — 1 for defaulted, 0 for repaid.

### Features

| Feature | Description |
|---|---|
| `AMT_INCOME_TOTAL` | Annual income (log-normal, $30K–$500K) |
| `AMT_CREDIT` | Loan credit amount ($50K–$1M) |
| `AMT_ANNUITY` | Monthly loan payment |
| `DAYS_BIRTH` | Applicant age in days (negative = days before application) |
| `DAYS_EMPLOYED` | Days employed (negative = currently employed) |
| `EXT_SOURCE_2` | External credit risk score (0–1, higher = lower risk) |
| `REGION_POPULATION_RELATIVE` | Normalised regional population density |
| `CODE_GENDER` | Gender (M / F) |
| `NAME_EDUCATION_TYPE` | Education level (4 categories) |
| `NAME_FAMILY_STATUS` | Marital status (5 categories) |

### Class Imbalance

The dataset has an **~8.6% default rate**, which is realistic for retail lending and creates a class imbalance problem. Both models handle this:
- **Logistic Regression**: `class_weight='balanced'`
- **XGBoost**: `scale_pos_weight` set to the ratio of negative to positive samples

---

## 3. Exploratory Data Analysis

**Plot: `01_eda_overview.png`** — Six-panel EDA grid

### What each panel shows

1. **Income Distribution by Default Status**
   - Defaulters (pink) are slightly skewed toward lower income bands
   - Higher income is a mild protective factor, consistent with the data-generating process

2. **External Risk Score (EXT_SOURCE_2) by Default**
   - The strongest visual separator: defaulters cluster at lower EXT_SOURCE scores
   - This feature is the dominant predictor in the model (confirmed by feature importance)

3. **Age Distribution by Default**
   - Younger applicants show marginally higher default rates
   - Age is a relatively weak signal on its own, but contributes in combination

4. **Default Rate by Gender**
   - Males have a slightly higher default rate (+3% probability in the simulation)
   - The difference is modest but statistically meaningful at N=10,000

5. **Default Rate by Education**
   - "Lower secondary" education correlates with highest default rates
   - "Higher education" applicants default least frequently

6. **Credit Amount vs Annuity Scatter**
   - Coloured by default (red = defaulted, green = repaid)
   - Shows the expected linear relationship between credit and annuity
   - No clear spatial cluster for defaults — defaults are spread across the credit spectrum

---

## 4. Data Preprocessing

Steps applied in sequence:

1. **Label Encoding** — Categorical columns (`CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`) are integer-encoded using `sklearn.LabelEncoder`
2. **Median Imputation** — `SimpleImputer(strategy='median')` fills any missing values (none in the simulated data, but the step is included for production robustness)
3. **Train/Test Split** — 80/20 split with `stratify=y` to preserve the default rate in both sets
4. **Standard Scaling** — `StandardScaler` applied to Logistic Regression inputs (XGBoost is tree-based and scale-invariant, so it uses unscaled features)

---

## 5. Model Building

Two classifiers are trained and compared:

### Logistic Regression (Baseline)
- `max_iter=1000`, `class_weight='balanced'`
- Linear decision boundary; interpretable coefficients
- Handles class imbalance via sample weighting

### XGBoost (Primary Model)
- `n_estimators=300`, `learning_rate=0.05`, `max_depth=5`
- Gradient boosted trees; captures non-linear interactions
- `scale_pos_weight` adjusts for class imbalance
- Evaluated with AUC metric during training

**Plot: `02_roc_curves.png`** — ROC curves for both models

The ROC curve plots True Positive Rate (sensitivity) vs False Positive Rate at every possible threshold. XGBoost achieves a meaningfully higher AUC (area under curve), indicating superior discrimination between defaulters and non-defaulters across all thresholds. The diagonal dashed line represents a random classifier (AUC = 0.5).

---

## 6. Feature Importance

**Plot: `03_feature_importance.png`** — XGBoost feature importance (horizontal bar chart)

XGBoost computes feature importance as the total reduction in the Gini impurity (or similar criterion) contributed by each feature across all trees. Key findings:

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `EXT_SOURCE_2` | External credit bureau score — strongest risk signal |
| 2 | `AMT_INCOME_TOTAL` | Income level — higher income = lower default risk |
| 3 | `AMT_ANNUITY` | Monthly repayment burden |
| 4 | `DAYS_BIRTH` | Applicant age — older applicants slightly safer |
| 5 | `AMT_CREDIT` | Total loan size |
| 6–10 | Categorical & employment features | Weaker but contributing signals |

---

## 7. Business Cost Optimization

This is the core contribution of the notebook beyond standard ML evaluation.

### The Problem with Threshold = 0.5

A classifier outputs a probability `p` that a loan will default. A threshold converts this into a binary decision:
- `p ≥ threshold` → **Reject** (predicted default)
- `p < threshold` → **Approve** (predicted repaid)

At threshold = 0.5, the model treats all errors equally. But in banking they are not — a missed default costs $5,000 while a wrongly rejected applicant costs only $1,200.

### Methodology

The notebook sweeps thresholds from 0.01 to 0.99 and computes total business cost at each point:

```
Total Cost = (False Negatives × $5,000) + (False Positives × $1,200)
```

The threshold that **minimises this cost function** is the optimal business threshold.

### Results

| Threshold | Description | Total Cost |
|---|---|---|
| 0.50 | Default (equal-error) | ~$999,200 |
| **0.78** | **Business-optimal** | **$857,200** |
| Savings | — | **$142,000** |

**Plot: `04_business_cost_optimization.png`** — Two-panel cost analysis

- **Left panel**: Total business cost as a function of threshold. The cost curve has a clear minimum around 0.78. The green dashed line marks the optimal point; the grey dotted line marks the naive 0.5.
- **Right panel**: The FP/FN trade-off. As the threshold increases, fewer loans are rejected (FPs fall) but more defaults slip through (FNs rise). The optimal threshold balances these curves weighted by their respective costs.

### Why is the Optimal Threshold Above 0.5?

Intuitively: because defaulters are a small minority (~8.6% of applicants), the model needs high confidence before flagging a loan as a default. A lower threshold would reject too many good customers, racking up FP costs. The $5,000 vs $1,200 asymmetry means we should be more willing to occasionally approve a borderline defaulter than to over-reject good customers en masse.

---

## 8. Results & Conclusions

**Plot: `05_confusion_matrices.png`** — Side-by-side confusion matrices

The left matrix (blue) shows XGBoost at the default 0.5 threshold. The right matrix (green) shows performance at the optimal 0.78 threshold.

### Summary of Findings

- **XGBoost significantly outperforms Logistic Regression** in terms of AUC on this imbalanced classification task
- **EXT_SOURCE_2 is the dominant risk factor** — a strong external credit score is the best single predictor of repayment
- **The optimal threshold (0.78) is conservative**: the model requires relatively high predicted default probability before rejecting an application, reflecting the reality that false positives (lost customers) accumulate quickly
- **Shifting from 0.5 to 0.78 saves the bank $142,000** on 2,000 test applicants — a ~14% cost reduction
- At scale, the savings would be proportionally larger

### Business Recommendation

Deploy XGBoost with the business-optimised threshold as the primary loan scoring system. Recalibrate the threshold quarterly as:
- Economic conditions change the default rate distribution
- The FN/FP cost ratio may shift with loan product mix
- New features or data sources become available

---

## 9. Generated Plots

| File | Content |
|---|---|
| `01_eda_overview.png` | 6-panel EDA: income, risk score, age, gender, education, credit scatter |
| `02_roc_curves.png` | ROC curves for Logistic Regression vs XGBoost with AUC scores |
| `03_feature_importance.png` | XGBoost feature importance bar chart |
| `04_business_cost_optimization.png` | Cost vs threshold curve + FP/FN trade-off panel |
| `05_confusion_matrices.png` | Confusion matrices at threshold 0.50 vs optimal threshold |

---

## 10. How to Run

```bash
# Clone / download the notebook
# Install dependencies (see below)
jupyter notebook Task4_Loan_Default_Risk.ipynb
# Run All Cells (Kernel → Restart & Run All)
```

The notebook is fully self-contained — it simulates its own dataset, so no external data download is required.

---

## 11. Dependencies

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
xgboost>=1.7
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

*Task 4 — DevelopersHub Corporation Data Science & Analytics Advanced Internship*
