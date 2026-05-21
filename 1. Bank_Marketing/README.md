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

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Notebook Walkthrough](#-notebook-walkthrough)
- [Model Performance](#-model-performance)
- [Explainable AI (SHAP)](#-explainable-ai-shap)
- [Key Insights](#-key-insights)
- [Business Recommendations](#-business-recommendations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔎 Project Overview

This project is a complete end-to-end **Data Science & Machine Learning pipeline** built as part of a Data Science Internship. The goal is to predict whether a bank customer will subscribe to a **term deposit** after being contacted during a phone-based marketing campaign.

The project covers every step of a professional ML workflow:

- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Training two classification models
- Evaluating models with multiple metrics
- Explaining predictions using **SHAP** (Explainable AI)
- Deriving actionable business insights

---

## 💼 Business Problem

Banks run large-scale **outbound phone marketing campaigns** to promote term deposits — a type of savings product where a customer locks in money for a fixed period at a guaranteed interest rate.

**The problem:** Calling every customer in the database is expensive, time-consuming, and annoying for customers who aren't interested. The bank needs a smarter approach.

**The solution:** A machine learning model that can **score each customer** and predict their likelihood of subscribing. This allows the marketing team to:

- Focus calls on high-probability customers
- Reduce campaign costs significantly
- Improve customer experience (less unsolicited contact)
- Increase the overall subscription conversion rate

> Even a small improvement in targeting can save thousands of agent hours and generate significant additional revenue per campaign cycle.

---

## 📂 Dataset Description

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)  
**File:** `bank-full.csv` (semicolon-separated)

| Property | Value |
|----------|-------|
| Total Records | 45,211 |
| Total Features | 16 input + 1 target |
| Target Variable | `y` (yes / no) |
| Subscription Rate | ~11.7% (class imbalance present) |
| Missing Values | None |

### 📊 Feature Dictionary

#### 👤 Customer Demographics

| Column | Type | Description |
|--------|------|-------------|
| `age` | Numeric | Age of the customer (years) |
| `job` | Categorical | Type of job (e.g. management, technician, retired, student) |
| `marital` | Categorical | Marital status (married / single / divorced) |
| `education` | Categorical | Education level (primary / secondary / tertiary / unknown) |
| `default` | Binary | Has credit in default? (yes / no) |
| `balance` | Numeric | Average yearly account balance in euros |
| `housing` | Binary | Has a housing loan? (yes / no) |
| `loan` | Binary | Has a personal loan? (yes / no) |

#### 📞 Last Contact Information

| Column | Type | Description |
|--------|------|-------------|
| `contact` | Categorical | Communication type (cellular / telephone / unknown) |
| `day` | Numeric | Last contact day of the month |
| `month` | Categorical | Last contact month of the year |
| `duration` | Numeric | Last contact call duration in seconds ⭐ (strongest predictor) |

#### 📈 Campaign Information

| Column | Type | Description |
|--------|------|-------------|
| `campaign` | Numeric | Number of contacts made during this campaign |
| `pdays` | Numeric | Days since the customer was last contacted from a previous campaign (-1 = never contacted) |
| `previous` | Numeric | Number of contacts made before this campaign |
| `poutcome` | Categorical | Outcome of the previous marketing campaign (success / failure / other / unknown) |

#### 🎯 Target Variable

| Column | Type | Description |
|--------|------|-------------|
| `y` | Binary | Did the customer subscribe? → **yes (1) / no (0)** |

---

## 📁 Project Structure

```
term-deposit-prediction/
│
├── 📓 Term_Deposit_Prediction.ipynb   ← Main Jupyter Notebook (complete project)
├── 📄 README.md                        ← This file
├── 📊 bank-full.csv                    ← Dataset (place in same directory)
│
├── 📈 Generated Plots (auto-saved when notebook runs)
│   ├── plot_01_target_distribution.png
│   ├── plot_02_age_distribution.png
│   ├── plot_03_job_subscription.png
│   ├── plot_04_marital_education.png
│   ├── plot_05_campaign_analysis.png
│   ├── plot_06_correlation_heatmap.png
│   ├── plot_07_monthly_trends.png
│   ├── plot_08_confusion_matrices.png
│   ├── plot_09_roc_curves.png
│   ├── plot_10_shap_importance.png
│   ├── plot_11_shap_summary.png
│   ├── plot_12_shap_waterfall.png
│   ├── plot_13_shap_force_5customers.png
│   ├── plot_14_feature_importance.png
│   ├── plot_15_age_group_analysis.png
│   └── plot_16_balance_analysis.png
│
└── requirements.txt                    ← Python dependencies
```

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `Python` | 3.10+ | Core programming language |
| `pandas` | 2.x | Data loading, manipulation, analysis |
| `numpy` | 1.x | Numerical computations |
| `matplotlib` | 3.x | Base plotting library |
| `seaborn` | 0.x | Statistical visualizations |
| `scikit-learn` | 1.x | ML models, preprocessing, evaluation |
| `shap` | 0.x | Model explainability (Shapley values) |
| `lime` | 0.x | Alternative local explainability |
| `nbformat` | 5.x | Notebook building utilities |
| `jupyter` | latest | Interactive notebook environment |

---

## ⚙️ Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/term-deposit-prediction.git
cd term-deposit-prediction
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap lime jupyter nbformat
```

### Step 4 — Add the Dataset

Download `bank-full.csv` from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and place it in the **same folder** as the notebook.

```
term-deposit-prediction/
├── Term_Deposit_Prediction.ipynb
└── bank-full.csv   ← place here
```

---

## ▶️ How to Run

### Option A — Jupyter Notebook (Recommended)

```bash
jupyter notebook Term_Deposit_Prediction.ipynb
```

Then in the browser: **Kernel → Restart & Run All**

### Option B — JupyterLab

```bash
jupyter lab Term_Deposit_Prediction.ipynb
```

### Option C — VS Code

Open the `.ipynb` file in VS Code with the **Jupyter extension** installed. Click **Run All**.

> ⚠️ **Important:** Make sure `bank-full.csv` is in the same working directory as the notebook before running. The notebook reads it with `pd.read_csv('bank-full.csv', sep=';')`.

---

## 📓 Notebook Walkthrough

The notebook is organized into **13 structured sections**, each with markdown explanations and complete Python code.

---

### Section 1 — Project Introduction
A detailed markdown introduction covering:
- What problem we're solving and why it matters
- An overview of the dataset columns and their business meaning
- The real-world impact of getting this prediction right

---

### Section 2 — Importing Libraries
All required libraries imported in one cell with clear comments explaining what each library is used for. Matplotlib style settings are also configured here for consistent, clean visuals throughout.

---

### Section 3 — Loading the Dataset
```python
df = pd.read_csv('bank-full.csv', sep=';')
```
The dataset uses semicolons as separators (not commas) — this is handled automatically. The section displays:
- First 5 rows (`df.head()`)
- All column names and data types
- Dataset shape and memory usage

---

### Section 4 — Data Exploration
A thorough exploration before touching the data:
- **Statistical summary** of all numerical columns (mean, std, min, max, quartiles)
- **Null value check** — confirmed zero missing values across all 45,211 rows
- **Duplicate check** — no duplicate rows found
- **Unique values** in all categorical columns
- **Target variable distribution** — 39,922 "no" (88.3%) vs 5,289 "yes" (11.7%)

> The dataset has a significant class imbalance which is addressed during model training using `class_weight='balanced'`.

---

### Section 5 — Exploratory Data Analysis (EDA)

**16 visualizations** across 7 plot groups:

#### 📊 Plot 1 — Target Distribution
A bar chart and pie chart showing the subscription rate. Clearly illustrates the class imbalance (~88% no, ~12% yes).

#### 📊 Plot 2 — Age Distribution
- Histogram comparing age distributions for subscribers vs non-subscribers
- Boxplot showing median ages per group
- Finding: Young adults (18–25) and seniors (65+) show the highest subscription tendency

#### 📊 Plot 3 — Job-wise Subscription Rate
Bar chart showing subscription rate (%) for each job type.  
Finding: **Retired and student** customers convert at the highest rate.

#### 📊 Plot 4 — Marital Status & Education
Side-by-side bar charts showing subscription rates by:
- Marital status (single customers convert slightly better)
- Education level (tertiary-educated customers convert better)

#### 📊 Plot 5 — Campaign Impact Analysis
- Histogram of number of contacts vs subscription outcome
- Boxplot of call duration by subscription outcome  
- Finding: **Longer calls** are strongly linked to subscriptions. **Fewer repeat calls** perform better.

#### 📊 Plot 6 — Correlation Heatmap
A lower-triangle heatmap of all numerical features showing pairwise correlations. `duration` shows the strongest positive correlation with the target variable.

#### 📊 Plot 7 — Monthly Subscription Trends
Bar chart of subscription rate by month (reordered chronologically).  
Finding: **March, September, October, and December** are peak subscription months.

---

### Section 6 — Data Preprocessing

Complete step-by-step encoding pipeline:

**Step 1 — Target Encoding**
```python
df_processed['y'] = df_processed['y'].map({'yes': 1, 'no': 0})
```

**Step 2 — Binary Column Encoding**  
Columns with only `yes`/`no` values (`default`, `housing`, `loan`) are mapped directly to `1`/`0`.

**Step 3 — One-Hot Encoding**  
Multi-class categorical columns are one-hot encoded using pandas `get_dummies()` with `drop_first=True` to avoid multicollinearity:
- `job` → 11 binary columns
- `marital` → 2 binary columns
- `education` → 3 binary columns
- `contact` → 2 binary columns
- `poutcome` → 3 binary columns
- `month` → 11 binary columns

After encoding: **52 total features**

**Step 4 — Feature Scaling**  
StandardScaler normalizes all features to zero mean and unit variance — required for Logistic Regression to converge properly.

---

### Section 7 — Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

| Split | Rows |
|-------|------|
| Training set | 36,168 samples (80%) |
| Testing set | 9,043 samples (20%) |

`stratify=y` ensures the 11.7% subscription rate is preserved in both splits.

---

### Section 8 — Model Building

#### 🔵 Model 1 — Logistic Regression

**What it is:** A linear model that estimates the probability of a binary outcome using a logistic (sigmoid) function.

**Why use it:**
- Fast to train, easy to interpret
- Works well as a baseline model
- Coefficients can be read directly as feature weights

**Configuration used:**
```python
LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
```
- `max_iter=1000` ensures convergence on a dataset with 52 features
- `class_weight='balanced'` compensates for the 88%/12% class imbalance

---

#### 🟢 Model 2 — Random Forest Classifier

**What it is:** An ensemble of many decision trees, each trained on a random subset of data and features. The final prediction is the majority vote across all trees.

**Why use it:**
- Handles non-linear relationships naturally
- Robust to outliers and irrelevant features
- Provides built-in feature importance rankings
- Less prone to overfitting than single decision trees

**Configuration used:**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
```
- `n_estimators=200` — 200 trees for stable, reliable predictions
- `max_depth=15` — prevents overfitting on deep trees
- `class_weight='balanced'` — handles class imbalance automatically
- `n_jobs=-1` — uses all available CPU cores for faster training

---

### Section 9 — Model Evaluation

Every model is evaluated using **6 metrics** to get a full picture of performance:

#### Accuracy Score
Percentage of correct predictions overall.

#### Confusion Matrix
Shows true positives, true negatives, false positives, and false negatives in a 2×2 grid. Visualized as color heatmaps for easy interpretation.

#### Classification Report
Precision, recall, and F1-score broken down per class:
- **Precision** — Of all customers predicted "yes", how many actually subscribed?
- **Recall** — Of all actual subscribers, how many did we correctly identify?
- **F1-Score** — Harmonic mean of precision and recall (most useful for imbalanced datasets)

#### ROC Curve & AUC Score
The ROC curve plots True Positive Rate vs False Positive Rate at all decision thresholds. The AUC (Area Under Curve) summarizes this into a single number:
- **AUC = 1.0** → perfect model
- **AUC = 0.5** → random guessing
- **AUC > 0.9** → excellent model

Both models are plotted on the same chart for direct comparison.

---

### Section 10 — Explainable AI with SHAP

**What is SHAP?**  
SHAP (SHapley Additive exPlanations) is based on game theory. It assigns each feature a "contribution score" for every individual prediction — telling us not just *what* the model predicted, but *why*.

**Why it matters in banking:**  
Financial institutions are legally and ethically required to explain credit and marketing decisions. A "black box" model isn't enough — we need to say "we didn't target this customer because their call duration was short and they had an outstanding loan."

#### 📊 SHAP Global Bar Plot (Plot 10)
Shows the **average absolute SHAP value** for each feature across all test customers. Higher = more important overall. `duration` dominates by a large margin.

#### 📊 SHAP Summary Dot Plot (Plot 11)
Each dot is one customer. Position on the x-axis shows the SHAP value (positive = pushes toward "yes", negative = pushes toward "no"). Color shows whether the original feature value was high (red) or low (blue).

**How to read it:**
- A red dot far to the right for `duration` means: *a customer with a long call duration was strongly pushed toward a "yes" prediction*
- A blue dot far to the left means: *a short call pushed the prediction toward "no"*

#### 📊 SHAP Waterfall Plot — Customer #1 (Plot 12)
A detailed breakdown for one individual customer. Starting from the baseline prediction (average model output), each feature either adds or subtracts from the final score. The waterfall chart makes the logic completely transparent.

#### 📊 5 Individual Prediction Explanations (Plot 13)
For each of 5 selected test customers, the notebook prints:
- Their actual label vs predicted label
- Their predicted subscription probability
- The top 5 features that influenced *their specific* prediction and the direction of influence

This is particularly valuable for compliance and auditing purposes.

---

### Section 11 — Customer Behavior Analysis

Synthesizing model results with data analysis:

- **Feature Importance Chart (Plot 14):** Top 15 features ranked by Random Forest importance score
- **Age Group Analysis (Plot 15):** Subscription rate broken down into 6 age bands (18–25, 26–35, 36–45, 46–55, 56–65, 65+)
- **Balance Distribution (Plot 16):** Account balance histograms for subscribers vs non-subscribers
- **Subscriber Profile Table:** Average age, balance, duration, campaign contacts, and previous contacts compared between the two groups

---

### Section 12 — Business Insights & Recommendations

Organized into four actionable strategy categories with specific, data-backed recommendations for the marketing team. See the [Business Recommendations](#-business-recommendations) section below.

---

### Section 13 — Final Conclusion

A complete summary table comparing both models, a list of key project findings, and suggestions for future improvements to take the project to a production-ready state.

---

## 📊 Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| **Accuracy** | 84.60% | 84.52% |
| **AUC-ROC** | 0.9079 | **0.9184** ✅ |
| **F1-Score (Weighted)** | ~0.84 | ~0.84 |
| **Training Speed** | Very fast | Moderate |
| **Interpretability** | High | Medium (needs SHAP) |

### 🏆 Best Model: Random Forest

Random Forest wins on **AUC-ROC** (0.9184 vs 0.9079), which is the most important metric for this use case. AUC measures how well the model separates subscribers from non-subscribers across all possible decision thresholds — crucial when you want to rank customers by likelihood and call only the top N%.

Both models score above **0.90 AUC**, meaning they are both excellent at distinguishing potential subscribers from the general population.

---

## 🔍 Explainable AI (SHAP)

Top 10 features by SHAP importance (Random Forest):

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | `duration` | Call length in seconds — longer calls = more engagement |
| 2 | `poutcome_success` | Previous campaign was a success — best predictor |
| 3 | `balance` | Higher account balance = more financially active customer |
| 4 | `age` | Age significantly influences financial product interest |
| 5 | `campaign` | Fewer calls this campaign = less saturation |
| 6 | `pdays` | Recently contacted from previous campaign |
| 7 | `previous` | Number of previous campaign contacts |
| 8 | `month_oct` | October contacts perform significantly better |
| 9 | `month_mar` | March contacts also outperform average |
| 10 | `job_retired` | Retired customers have highest subscription rates |

**Key SHAP Insight:** `duration` is far and away the most important feature. However, this has an important caveat — you can't know the call duration *before* making the call. This means `duration` is a strong *post-hoc* indicator but should not be used for *pre-campaign* targeting. The next most actionable features are `poutcome_success`, `balance`, and `age`.

---

## 💡 Key Insights

### 📞 About the Campaign
- Customers contacted **1–2 times** in a campaign convert far better than those contacted 5+ times
- **March, September, October, and December** campaigns have the highest subscription rates
- **Longer calls** strongly indicate customer interest — calls under 60 seconds almost never convert

### 👥 About the Customers
- **Retired and student** customers have the highest subscription rates across all job types
- **Young adults (18–25)** and **seniors (65+)** are the most responsive age groups
- Customers who **successfully subscribed in a previous campaign** are highly likely to subscribe again
- Customers **without housing loans** are better prospects — they have more financial flexibility

### 💰 About Financials
- Subscribers have **higher average account balances** than non-subscribers
- Customers in **credit default** rarely subscribe — they have more pressing financial concerns
- Customers with **no personal loans** respond more positively to the campaign

---

## 📣 Business Recommendations

### 🎯 Customer Targeting
- Build a **pre-campaign scoring model** using all features *except* `duration` (not available pre-call)
- Focus outreach on retired, student, and young adult segments
- Prioritize customers with a **history of previous successful subscriptions** (`poutcome = success`)

### 📞 Campaign Execution
- **Cap contacts at 3 per customer per campaign** — more calls than this drastically reduces conversion
- Schedule campaigns in **March, September, October, and December** for maximum impact
- Brief agents to aim for **quality conversations** — the duration-subscription correlation suggests engaged discussions convert better

### 💡 Resource Allocation
- Use the model to **score all customers** before any calls are made
- Only contact the **top 25–30% probability** customers — this will capture most potential subscribers while drastically reducing total call volume
- Reallocate saved agent hours toward follow-up conversations with high-probability leads

### 📊 Ongoing Improvement
- **Retrain the model quarterly** as customer behavior and campaign data evolves
- Track model performance post-deployment with a feedback loop from actual outcomes
- Consider A/B testing: model-driven targeting vs random sampling to measure real business impact

---

## 🚀 Future Improvements

| Improvement | Expected Benefit |
|-------------|-----------------|
| **XGBoost / LightGBM** | Likely 2–4% AUC improvement with better handling of feature interactions |
| **SMOTE oversampling** | Better recall on minority class (actual subscribers) by synthetically balancing training data |
| **Cross-validation (k-fold)** | More reliable performance estimates, reduces variance from single train-test split |
| **Hyperparameter tuning (GridSearchCV)** | Potentially 1–3% accuracy improvement by optimizing tree depth, estimators, etc. |
| **Flask / FastAPI deployment** | Real-time scoring API that campaign agents can query before each call |
| **Feature selection** | Remove low-importance features to reduce overfitting and improve inference speed |
| **Threshold optimization** | Tune the decision threshold (default 0.5) to maximize F1 or precision/recall balance for business needs |

---

## 👤 Author

**Mirza Muhammad Ahsan**  
Data Science Intern  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)  
💻 [GitHub](https://github.com/mirzaahsan143)

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository — Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- Original paper: Moro et al., 2014 — *"A Data-Driven Approach to Predict the Success of Bank Telemarketing"*
- SHAP library: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)

---

<p align="center">
  ⭐ If you found this project helpful, please give it a star on GitHub!
</p>
