# 🏦 Term Deposit Subscription Prediction
### Bank Marketing Campaign — Binary Classification with Explainable AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

---

## 📌 What Is This Project?

Banks run phone-based marketing campaigns to convince customers to open a **term deposit** — a savings product where money is locked in for a fixed period at a guaranteed interest rate. Calling every single customer in the database is expensive, time-consuming, and frustrating for customers who have no interest.

This project builds a **machine learning model that predicts which customers are likely to subscribe** before any call is made. The bank can then focus its campaign resources on high-probability leads, cutting costs while increasing conversions.

> **Task Type:** Binary Classification — predict `yes` (will subscribe) or `no` (will not subscribe)

---

## 📂 Dataset

**Source:** UCI Machine Learning Repository — Bank Marketing Dataset  
**File:** `bank-full.csv` (semicolon-separated)

| Property | Detail |
|----------|--------|
| Total Records | 45,211 customer entries |
| Features | 16 input features + 1 target variable |
| Target Variable | `y` → yes / no |
| Subscription Rate | ~11.7% (highly imbalanced — most customers say no) |
| Missing Values | None |

### Feature Breakdown

**Customer Profile**

| Feature | Description |
|---------|-------------|
| `age` | Customer's age in years |
| `job` | Job type — management, technician, retired, student, etc. |
| `marital` | Marital status — married, single, divorced |
| `education` | Highest education level — primary, secondary, tertiary |
| `default` | Has credit in default? |
| `balance` | Average yearly account balance (€) |
| `housing` | Has a housing loan? |
| `loan` | Has a personal loan? |

**Campaign & Contact Details**

| Feature | Description |
|---------|-------------|
| `contact` | How the customer was contacted — cellular, telephone |
| `day` / `month` | When the last contact was made |
| `duration` | How long the last call lasted (seconds) — strongest predictor |
| `campaign` | Number of times contacted during this campaign |
| `pdays` | Days since last contact from a previous campaign |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of the previous campaign — success, failure, unknown |

---

## 🔍 What Happens in the Notebook

### 1. Data Exploration
Before building anything, the dataset is thoroughly examined — checking data types, distributions, class balance, and unique values in every column. The target variable turns out to be heavily skewed: **88.3% of customers said no**, only **11.7% said yes**. This class imbalance is a key challenge that influences every decision downstream.

### 2. Exploratory Data Analysis (EDA)
Sixteen visualizations are created to understand patterns before touching the models:

- **Age distribution** — subscribers skew toward younger (18–25) and older (65+) age groups
- **Job analysis** — retired and student customers convert at the highest rates
- **Campaign impact** — fewer contacts per customer leads to better outcomes; customers contacted 5+ times rarely subscribe
- **Call duration** — the single most telling signal; long calls strongly associate with eventual subscription
- **Monthly trends** — March, September, October, and December are peak months for subscriptions
- **Correlation heatmap** — `duration`, `pdays`, and `previous` show the strongest numerical relationships with the target

### 3. Data Preprocessing
All categorical columns are converted to numbers so the models can process them:

- Binary columns (`default`, `housing`, `loan`) are mapped directly to 0/1
- Multi-class columns (`job`, `marital`, `education`, `contact`, `poutcome`, `month`) are one-hot encoded — each category becomes its own binary column
- The target variable (`y`) is encoded as `yes → 1`, `no → 0`
- All features are standardized using **StandardScaler** so Logistic Regression converges properly

After encoding the dataset expands from 16 to **52 features**.

### 4. Model Training
Two classification models are trained and compared:

**Logistic Regression**
A linear model that calculates the probability of subscription using a sigmoid function. It's fast, interpretable, and serves as a strong baseline. `class_weight='balanced'` is applied so the model doesn't ignore the minority class (subscribers).

**Random Forest Classifier**
An ensemble of 200 decision trees, each trained on a random subset of data and features. The final prediction is the majority vote. It naturally captures non-linear relationships and complex feature interactions that Logistic Regression misses. Also trained with `class_weight='balanced'`.

### 5. Model Evaluation
Both models are measured across six metrics:

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | Overall % of correct predictions |
| **Precision** | Of predicted subscribers, how many actually subscribed? |
| **Recall** | Of actual subscribers, how many did the model catch? |
| **F1-Score** | Balance between precision and recall — most useful for imbalanced data |
| **Confusion Matrix** | Breakdown of true/false positives and negatives |
| **AUC-ROC** | Model's ability to rank subscribers above non-subscribers at all thresholds |

The **ROC curves for both models are plotted on the same chart** for direct visual comparison, along with side-by-side confusion matrix heatmaps.

### 6. Explainable AI — SHAP
After evaluating performance, the notebook goes deeper: **why** is the Random Forest making each prediction? SHAP (SHapley Additive exPlanations) answers this using game theory — it assigns every feature a contribution score for every single prediction.

Four types of SHAP output are generated:
- **Global bar chart** — overall feature importance across all customers
- **Summary dot plot** — shows both the importance and direction of each feature's impact
- **Waterfall chart** — breaks down one individual customer's prediction step by step
- **5 individual explanations** — five specific customers with their actual vs predicted label, subscription probability, and the top features that drove their result

### 7. Customer Behavior Analysis
Using both SHAP values and raw data analysis, the notebook identifies which types of customers are most likely to subscribe and what factors drive that decision. Average subscriber profiles are compared against non-subscribers across age, balance, call duration, and campaign intensity.

---

## 📊 Results

| Model | Accuracy | AUC-ROC | F1-Score |
|-------|----------|---------|---------|
| Logistic Regression | 84.60% | 0.9079 | ~0.84 |
| **Random Forest** | **84.52%** | **0.9184** ✅ | **~0.84** |

**Random Forest is the better model.** While accuracy is nearly identical, its AUC-ROC score of **0.9184** means it does a significantly better job of ranking customers by their true likelihood to subscribe — which is exactly what a real campaign targeting system needs.

An AUC above 0.90 is considered excellent. Both models clear this bar, confirming the features in this dataset carry genuine predictive signal.

---

## 🔍 Top Predictive Features (SHAP)

| Rank | Feature | Why It Matters |
|------|---------|---------------|
| 1 | `duration` | Longer calls mean the customer was engaged and interested |
| 2 | `poutcome_success` | Previously said yes → most likely to say yes again |
| 3 | `balance` | Higher savings balance = more financially active customer |
| 4 | `age` | Age group significantly shapes financial product interest |
| 5 | `campaign` | Fewer calls this round = less saturated, more receptive |
| 6 | `pdays` | Recently contacted from a past campaign, still warm |
| 7 | `month_oct` | October campaigns dramatically outperform the average |
| 8 | `month_mar` | March is the second-best month for subscriptions |
| 9 | `job_retired` | Retired customers have the highest conversion rate of any job type |
| 10 | `previous` | More prior campaign contacts builds familiarity |

> **Important caveat on `duration`:** Call length is the strongest predictor, but it's only known *after* the call ends — it can't be used to decide *who to call* before a campaign. The next most actionable features for pre-campaign targeting are `poutcome_success`, `balance`, and `age`.

---

## 💡 Key Findings

**About customers:**
- Retired and student customers have the highest subscription rates across all job types
- Young adults (18–25) and seniors (65+) respond better than middle-aged groups
- Customers who subscribed in a previous campaign are by far the most likely to subscribe again
- Customers without housing or personal loans are significantly better prospects — they have fewer financial burdens

**About the campaign:**
- Calling a customer more than 3 times in a single campaign sharply reduces conversion rates — over-contacting kills interest
- The best months to run campaigns are March, September, October, and December
- Customers with higher account balances are more receptive — they have the financial means to lock money away

**About the model:**
- The Random Forest model correctly identifies the majority of actual subscribers while keeping false alarms manageable
- SHAP confirms the model's reasoning is logically sound — it's learning genuine behavioral patterns, not noise
- The model can be used to score all customers before a campaign and rank them by subscription probability

---

## 💼 Business Recommendations

**Who to target:**
Focus on retired customers, students, young adults, and seniors. Prioritize anyone who successfully subscribed in a previous campaign — they are the most reliable segment. Filter for customers with positive account balances and no outstanding personal loans.

**How to run the campaign:**
Limit contact attempts to a maximum of 3 per customer. Plan campaign pushes around March, September, October, and December. Train agents to have longer, quality conversations rather than rushing through call lists — engagement during the call is the strongest signal of eventual conversion.

**How to use the model:**
Score every customer in the database before the campaign begins. Only contact the top 25–30% by predicted probability. This approach captures the vast majority of potential subscribers while eliminating the majority of wasted calls — dramatically improving cost efficiency without sacrificing revenue.

---

## 🚀 What Could Be Improved

| Improvement | Why It Helps |
|-------------|-------------|
| XGBoost / LightGBM | Likely 2–4% AUC gain through better handling of feature interactions |
| SMOTE oversampling | Improves recall on actual subscribers by synthetically balancing the training data |
| Cross-validation | More reliable performance estimates across multiple data splits |
| Hyperparameter tuning | GridSearchCV could squeeze additional performance from both models |
| Remove `duration` for pre-call scoring | Makes the model truly deployable before campaigns run |
| Threshold optimization | Tune the 0.5 decision cutoff to match the business's tolerance for false positives |

---

## 🛠 Libraries Used

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `shap` · `lime` · `jupyter`

---

*Data Science Internship Project · UCI Bank Marketing Dataset · Binary Classification · Explainable AI*
