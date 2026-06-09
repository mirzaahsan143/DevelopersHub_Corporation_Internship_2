<div align="center">

# 🚀 DevelopersHub Corporation
## Data Science & Analytics Internship — Phase 2

### By **Mirza Muhammad Ahsan**

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-FF6600?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

<br>

*5 end-to-end Data Science & Analytics projects spanning Classification, Clustering, Time Series Forecasting, Risk Analytics, and Business Intelligence*

</div>

---

## 📖 Internship Overview

This repository documents **five production-grade data science projects** completed during the DevelopersHub Corporation Advanced Data Science & Analytics Internship (Phase 2). Each project addresses a distinct real-world business problem — from predicting bank marketing outcomes to forecasting household energy consumption to building an interactive retail BI dashboard.

Across the five tasks, the work demonstrates the full analytics lifecycle: raw data ingestion, cleaning and feature engineering, exploratory analysis, machine learning model development, business-cost-aware evaluation, and stakeholder-ready visualization and reporting. Every project prioritises **business value over academic metrics** — findings are framed in terms of revenue impact, cost reduction, and actionable strategy rather than benchmark scores alone.

---

## 🧠 Skills Demonstrated

| Category | Skills & Techniques |
|---|---|
| **Data Cleaning** | Missing-value imputation (ffill/bfill, median), outlier clipping, datetime engineering, deduplication |
| **EDA** | Distribution analysis, correlation heatmaps, group comparisons, temporal trend decomposition |
| **Machine Learning** | Logistic Regression, Random Forest, XGBoost, K-Means Clustering |
| **Time Series Forecasting** | ARIMA, Facebook Prophet, XGBoost with lag features, stationarity testing (ADF) |
| **Clustering** | K-Means, Elbow Method, Silhouette Score, PCA, t-SNE |
| **Business Intelligence** | KPI design, interactive dashboards, data storytelling, cohort analysis |
| **Data Visualization** | Matplotlib, Seaborn, Plotly (interactive), Streamlit |
| **Model Evaluation** | Accuracy, AUC-ROC, F1, Precision/Recall, Confusion Matrix, SHAP explainability |
| **Business Recommendations** | Threshold optimisation, cost-matrix analysis, segment-level marketing strategy |

---

## 📌 Task 1 — Bank Marketing Campaign Prediction

### 🧩 Problem Statement
A Portuguese bank runs phone-based campaigns to sell term deposit products. Calling every customer is costly and ineffective. The goal is to predict **which customers will subscribe** before the first call is made — enabling the bank to focus resources on high-probability leads.

### 📂 Dataset
**UCI Bank Marketing Dataset** — `bank-full.csv`

| Property | Detail |
|---|---|
| Records | 45,211 customer entries |
| Features | 16 input features + 1 binary target (`y`) |
| Subscription Rate | ~11.7% — heavily imbalanced |
| Missing Values | None |

Features span customer demographics (age, job, marital status, education, account balance), campaign history (contacts made, previous outcome, call duration), and contact logistics (month, day, contact type).

### ⚙️ Data Cleaning & Preprocessing
- Binary columns (`default`, `housing`, `loan`) mapped directly to 0/1
- Multi-class categoricals one-hot encoded — dataset expanded from 16 to **52 features**
- Target encoded: `yes → 1`, `no → 0`
- All features standardised with `StandardScaler` for Logistic Regression convergence
- Class imbalance handled via `class_weight='balanced'` on both models

### 🔍 EDA Highlights
- **Retired and student** customers show the highest subscription conversion rates
- **Call duration** is the single strongest predictor — longer calls signal genuine engagement
- Customers contacted **more than 3 times** in a campaign have sharply lower conversion
- **March, September, October, and December** are the best months for campaigns
- Customers with **prior successful campaign outcomes** are far more likely to subscribe again

### 🤖 Models Used

| Model | Accuracy | AUC-ROC | F1-Score |
|---|---|---|---|
| Logistic Regression | 84.60% | 0.9079 | ~0.84 |
| **Random Forest** ✅ | **84.52%** | **0.9184** | **~0.84** |

Random Forest is selected as the production model — its AUC of **0.9184** (excellent threshold) means it reliably ranks subscribers above non-subscribers across all decision thresholds.

### 🔬 SHAP Explainability
SHAP (SHapley Additive exPlanations) was applied post-training to explain *why* the model makes each prediction. Top drivers:

| Rank | Feature | Business Meaning |
|---|---|---|
| 1 | `duration` | Engagement proxy — long calls → genuinely interested customer |
| 2 | `poutcome_success` | Best leading indicator: if they said yes before, they likely will again |
| 3 | `balance` | Financially active customers have the means to lock money away |
| 4 | `age` | Retirement-age and young adults respond to term deposits most |
| 5 | `campaign` | Fewer calls this round = less fatigued, more receptive |

> ⚠️ `duration` is only known *after* a call ends — it cannot be used for pre-campaign targeting. The next most actionable features are `poutcome_success`, `balance`, and `age`.

### 💡 Business Recommendations
1. **Prioritise past subscribers** — they are the single highest-conversion segment
2. **Score all customers before dialling** — contact only the top 25–30% by predicted probability
3. **Cap contact attempts at 3 per campaign** — beyond that, conversion collapses and customer goodwill erodes
4. **Focus campaign spend on March, September–October, December** — avoid summer months
5. **Target retired and student segments** — they outperform every other job category

### 🏁 Outcome
A deployable customer scoring pipeline that, when applied to the full database before any calls are made, enables the bank to capture the vast majority of potential subscribers while eliminating the majority of wasted calls — improving campaign ROI significantly.

---

## 📌 Task 2 — Customer Segmentation using K-Means Clustering

### 🧩 Problem Statement
A mall's marketing team was applying the same promotions to all customers, leaving high-value segments under-served and low-value segments over-marketed to. The goal is to **segment the customer base into distinct behavioural groups** so each segment receives a tailored, relevant strategy.

### 📂 Dataset
**Mall Customers Dataset** — `Mall_Customers.csv`

| Property | Detail |
|---|---|
| Records | 200 customers |
| Features | Age, Gender, Annual Income (k$), Spending Score (1–100) |
| Missing Values | None |
| Duplicates | None |

### ⚙️ Data Preparation
- `CustomerID` dropped (no behavioural signal)
- Column names standardised (`Genre → Gender`, `Annual Income (k$) → Annual_Income`)
- `StandardScaler` applied to all numeric features — critical for Euclidean distance-based K-Means
- Separate 2D (Income + Spending) and 3D (Age + Income + Spending) feature sets used for interpretability vs. depth

### 🤖 Techniques Used
- **Elbow Method** (WCSS inertia curve) + **Silhouette Score** — both confirmed **K=5** as optimal
- **K-Means Clustering** — 5-cluster solution on 2D and 3D feature spaces
- **PCA** — dimensionality reduction to 2D components for cluster validation
- **t-SNE** — non-linear embedding to confirm tight, well-separated cluster structure

### 📊 The 5 Customer Segments

| Cluster | Profile | Income | Spending | Strategy |
|---|---|---|---|---|
| 0 — Sensible Savers | Middle income, moderate spenders | Medium | Medium | Loyalty points, cashback, value bundles |
| 1 — Premium Customers 💎 | High income, high spenders | High | High | VIP memberships, exclusive events, luxury experiences |
| 2 — Budget Shoppers | Low income, low spenders | Low | Low | Discount coupons, clearance sales, daily deals |
| 3 — Careless Spenders ⭐ | Low income, high spenders (impulsive) | Low | High | BNPL options, limited-edition drops, influencer campaigns |
| 4 — Conservative Elites 💰 | High income, low spenders (untapped) | High | Low | Premium consultation, exclusive experiences — avoid discounts |

### 💡 Marketing Recommendations
- **Cluster 1 (Premium)** is the highest lifetime-value group — invest in retention, not acquisition discounts
- **Cluster 4 (Conservative Elites)** is the biggest missed opportunity — they have spending power but aren't using it; personalised advisory services unlock this segment
- **Cluster 3 (Careless Spenders)** are impulse buyers — time-limited offers and BNPL dramatically increase basket size
- Avoid applying blanket discount campaigns; they erode margin with Cluster 1 and Cluster 4 without adding value

### 🏁 Outcome
A fully validated 5-segment customer taxonomy with concrete, segment-specific marketing playbooks ready for campaign planning.

---

## 📌 Task 3 — Energy Consumption Forecasting

### 🧩 Problem Statement
Accurate short-term energy forecasting is critical for grid stability, demand-response programmes, and smart home automation. This project builds a forecasting pipeline on minute-resolution household power telemetry — identifying patterns and predicting future consumption to enable cost savings of an estimated **8–15% annually** on dynamic tariffs.

### 📂 Dataset
**UCI Household Power Consumption Dataset** — `household_power_consumption.csv`

| Property | Detail |
|---|---|
| Records | 260,640 minute-resolution readings |
| Period | January–June 2007 |
| Location | Sceaux, France |
| Target | `Global_active_power` (kW) |
| Missing Values | 3,771 records (1.45%) |

Sub-metering splits consumption across three circuits: Kitchen (SM1), Laundry (SM2), Water Heater & AC (SM3).

### ⚙️ Data Cleaning & Feature Engineering
- `Date` + `Time` concatenated into a continuous `Datetime` index
- Gaps ≤60 minutes imputed with `ffill`; remaining trailing gaps filled with `bfill`
- Impossible negative active-power readings clipped to 0 kW
- Engineered features: `Sub_metering_total`, `Active_power_wh` (kW → Wh/min), `Energy_remaining` (unmetered consumption), plus temporal features (hour, day of week, month, is_weekend)

### 📈 Time Series Analysis
- Target distribution is heavily right-skewed — CV of **102.3%**; mean (1.15 kW) far exceeds median (0.53 kW)
- `Global_active_power` and `Global_intensity` are near-perfectly correlated (r ≈ 0.99) — Ohm's Law confirmed
- `Sub_metering_3` (Water Heater/AC) is the primary demand spike driver (mean 5.74 Wh, σ 8.15 Wh)
- **ADF Test** on daily aggregates: statistic −2.80, p-value 0.058 → confirmed non-stationarity → first-order differencing (d=1) required for ARIMA

### 🤖 Models Used

| Model | Architecture | Key Finding |
|---|---|---|
| **ARIMA** | Univariate statistical baseline | ADF-guided d=1 differencing; captures autocorrelation and moving average components |
| **Facebook Prophet** | Generalised Additive Model | Strong at weekly seasonality and trend; under-predicted extreme evening spikes |
| **XGBoost** ✅ | Gradient Boosted Trees | Best handling of skewed distribution; heavy regularisation prevented overfitting to sensor noise |

### 💡 Business Recommendations
1. **Smart scheduling**: shift high-load appliances (washing machine, dishwasher) to off-peak hours identified by the forecast
2. **Anomaly detection**: flag when real-time consumption deviates >2σ from forecast — indicator of energy theft or appliance degradation
3. **Dynamic tariff optimisation**: households on time-of-use pricing can automate appliance scheduling against the forecast curve to achieve 8–15% cost reduction
4. **Utility demand response**: aggregate household-level forecasts into grid-level demand curves for proactive supply balancing

### 🏁 Outcome
A multi-model forecasting pipeline that processes raw minute-level telemetry into actionable predictions, with a clear recommendation to deploy XGBoost for production use given its superior handling of the highly skewed power distribution.

---

## 📌 Task 4 — Loan Default Risk Prediction

### 🧩 Problem Statement
Banks face asymmetric financial consequences when making lending decisions: approving a loan that defaults costs **$5,000**, while wrongly rejecting a creditworthy applicant costs **$1,200** in lost revenue. Standard ML models optimised at threshold 0.5 treat both errors equally — this project builds a pipeline that finds the **decision threshold that minimises total business cost**.

### 📂 Dataset
Simulated **Home Credit-style** lending dataset — 10,000 applicants, seed = 42

| Property | Detail |
|---|---|
| Records | 10,000 applicants |
| Default Rate | ~8.6% (realistic for retail lending) |
| Features | 10 financial and demographic features |
| Target | `TARGET` (1 = defaulted, 0 = repaid) |

Key features: Annual Income, Credit Amount, Monthly Annuity, Age (`DAYS_BIRTH`), Employment tenure, External credit score (`EXT_SOURCE_2`), Gender, Education, Family status.

### ⚙️ Data Preparation
- Categorical features label-encoded (`CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`)
- Median imputation applied for production robustness
- 80/20 stratified train/test split preserving default rate
- StandardScaler applied for Logistic Regression; XGBoost uses raw features (tree models are scale-invariant)
- Class imbalance addressed: `class_weight='balanced'` for LR; `scale_pos_weight` for XGBoost

### 🤖 Models Used

| Model | AUC-ROC | Notes |
|---|---|---|
| Logistic Regression (baseline) | ~0.76 | Linear boundary; interpretable |
| **XGBoost** ✅ | **~0.82** | Non-linear; handles imbalance; deployed model |

### 💸 Cost Optimisation Strategy

The notebook sweeps all thresholds from 0.01 to 0.99, computing:

```
Total Cost = (False Negatives × $5,000) + (False Positives × $1,200)
```

| Threshold | Total Cost | Notes |
|---|---|---|
| 0.50 | ~$999,200 | Naive equal-error default |
| **0.78** | **$857,200** ✅ | **Business-optimal** |
| **Savings** | **$142,000** | ~14% cost reduction on 2,000 test cases |

The optimal threshold (0.78) is above 0.5 because the ~8.6% minority class means the model needs high confidence before flagging a default — over-rejection of good customers accumulates FP costs faster than it saves on FN costs.

### 💡 Business Recommendations
1. **Deploy XGBoost at threshold 0.78** as the primary loan scoring system
2. **Monitor EXT_SOURCE_2** — the single strongest risk predictor; ensure the credit bureau feed is live and up to date
3. **Recalibrate threshold quarterly** — economic cycles shift default rates and the optimal cost balance
4. **Never use a fixed 0.5 threshold** in asymmetric-cost lending problems; the $142K saving on a 2,000-applicant test set scales to millions annually
5. **Extend with bureau data** — adding payment history, credit utilisation, and derogatory marks would materially improve AUC

### 🏁 Outcome
A production-ready lending risk pipeline with an explicitly business-cost-optimised decision threshold, delivering a 14% reduction in total financial loss versus the naive equal-error approach.

---

## 📌 Task 5 — Global Superstore Business Intelligence Dashboard

### 🧩 Problem Statement
Global Superstore generates $12.64M in annual revenue across 147 countries but lacked a centralised analytics system. Management had no real-time visibility into which regions, product lines, and customer segments were driving or destroying profitability. This project delivers a full BI solution.

### 📂 Dataset
**Global Superstore Dataset** — `Global_Superstore.xls`

| Property | Detail |
|---|---|
| Records | 51,290 transactions |
| Features | 24 columns |
| Time Period | Jan 2011 – Dec 2014 |
| Markets | 7 global markets (US, EU, APAC, LATAM, EMEA, Africa, Canada) |
| Customers | 795 unique |
| Categories | 3 (Technology, Furniture, Office Supplies) |

### ⚙️ Data Cleaning
- Date columns parsed and validated for temporal integrity
- Duplicate order-row entries resolved
- Negative profit values preserved (real loss-making orders, not data errors)
- Derived metrics: Profit Margin %, Average Order Value, Customer Lifetime Value proxy

### 📊 EDA Highlights
- **$12.64M** total revenue with consistent YoY growth; **Q4 is 35% above Q1** every year
- **APAC and EU** are the dominant and most profitable markets
- **Discounts above 30%** consistently produce negative average profit per order
- **Tables and Bookcases** sub-categories are chronically loss-making — structural pricing issues
- **Top 10% of customers generate ~50% of revenue** — classic Pareto distribution
- **Technology (37% of sales)** is the highest-revenue and highest-margin category

### 🖥️ Dashboard Features (Built with Streamlit + Plotly)

**Sidebar Filters:** Year (2011–2014) · Region (13) · Category · Sub-Category (cascading)

**KPI Cards:** Total Sales · Total Profit · Total Orders · Unique Customers · Average Order Value · Profit Margin %

**14 Interactive Charts:**
- Monthly Sales & Profit trends (area + colour-coded bar)
- Yearly Sales & Profit comparison
- Sales & Profit by Region, Category, Sub-Category, and Segment
- Segment × Category heatmap
- Top 10 Customers and Top 10 Products
- Sales vs Profit bubble chart by market
- Discount Rate vs Profit scatter (key finding visualised)
- Ship Mode distribution

**Automated Data Storytelling:** 8 dynamic business insights generated from the current filter selection — updated on every filter change.

### 💡 Strategic Recommendations
1. **Eliminate or reprice Tables and Bookcases** — they destroy margin at scale; no volume justifies sustained losses
2. **Cap discount authority at 30%** — above this, every order is margin-negative; tie discounts to approval workflows
3. **Invest in APAC and EU retention** — highest revenue and margin; customer churn here has outsized impact
4. **Increase focus on Home Office segment** — smallest by volume but highest margin per order
5. **Run Q4 campaigns earlier** — start inventory buildup and targeted outreach in August to capture the October–December surge
6. **Reward the top 20% of customers** — they generate 80% of revenue; a formal loyalty programme would protect this concentration

### 🏁 Outcome
A fully interactive, production-ready BI dashboard deployable via `streamlit run app.py` — transforming 51,290 raw transactions into a real-time decision support tool for management across all business dimensions.

---

## 📊 Cross-Project Comparison

| Task | Domain | Dataset Size | Core Technique | Primary Business Goal |
|---|---|---|---|---|
| 1 — Bank Marketing | Financial Services | 45,211 records | Binary Classification + SHAP | Reduce campaign cost, increase subscription conversion |
| 2 — Customer Segmentation | Retail | 200 customers | K-Means Clustering + PCA/t-SNE | Enable targeted, personalised marketing by segment |
| 3 — Energy Forecasting | Utilities / Smart Home | 260,640 readings | Time Series: ARIMA + Prophet + XGBoost | Predict demand; enable dynamic tariff savings of 8–15% |
| 4 — Loan Default Risk | Banking / Credit | 10,000 applicants | XGBoost + Cost-Threshold Optimisation | Minimise total lending loss through optimal decision threshold |
| 5 — Global Superstore BI | Retail / E-Commerce | 51,290 transactions | EDA + Interactive BI Dashboard | Centralise KPI monitoring; identify margin leaks and growth levers |

---

## 🏆 Key Internship Achievements

- 🤖 **4 machine learning models** built, evaluated, and interpreted with business-cost framing
- 📈 **3 time series forecasting architectures** compared on real-world energy telemetry
- 🗂️ **5 customer segments** identified with concrete, segment-specific marketing playbooks
- 💸 **$142,000 business saving** demonstrated on loan default cost optimisation
- 🖥️ **1 production Streamlit BI dashboard** with 14 interactive charts and automated insights
- 🔍 **SHAP explainability** applied to communicate model decisions to non-technical stakeholders
- 📊 **50+ visualisations** produced across all 5 projects
- 🔁 **End-to-end analytics workflows** from raw data to business recommendation in every task

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4EABE6?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square)
![Statsmodels](https://img.shields.io/badge/Statsmodels-306998?style=flat-square)
![Prophet](https://img.shields.io/badge/Prophet-0467DF?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 📁 Repository Structure

```
DevelopersHub_Corporation_Internship_2/
│
├── 1. Bank_Marketing/
│   ├── Term_Deposit_Prediction.ipynb     # Full notebook with SHAP
│   ├── bank-full.csv                      # UCI Bank Marketing dataset
│   ├── README.md
│   └── Output plots/                      # 9 EDA & model evaluation charts
│       ├── plot_01_target_distribution.png
│       ├── plot_02_age_distribution.png
│       ├── plot_03_job_subscription.png
│       ├── plot_04_marital_education.png
│       ├── plot_05_campaign_analysis.png
│       ├── plot_06_correlation_heatmap.png
│       ├── plot_07_monthly_trends.png
│       ├── plot_08_confusion_matrices.png
│       └── plot_09_roc_curves.png
│
├── 2. Customer Segmentation/
│   ├── Customer_Segmentation_Project.ipynb
│   ├── Mall_Customers.csv
│   ├── README.md
│   └── Output Graphs/                     # 15 clustering & EDA charts
│       ├── elbow_method.png
│       ├── silhouette_scores.png
│       ├── kmeans_clusters_2d.png
│       ├── pca_clusters.png
│       ├── tsne_clusters.png
│       └── ...
│
├── 3. Energy Consumption/
│   ├── Energy_Forecasting_Notebook.ipynb
│   ├── household_power_consumption.csv    # UCI dataset (260K records)
│   ├── README.md
│   └── Output Graphs/                     # 13 time series & model charts
│       ├── resampling.png
│       ├── daily_rolling.png
│       ├── acf_pacf.png
│       ├── arima_forecast.png
│       ├── prophet_components.png
│       ├── model_comparison.png
│       └── ...
│
├── 4. Loan Default Risk/
│   ├── Task4_Loan_Default_Risk.ipynb
│   ├── README.md
│   └── Output Plots/                      # 5 risk & cost optimisation charts
│       ├── 01_eda_overview.png
│       ├── 02_roc_curves.png
│       ├── 03_feature_importance.png
│       ├── 04_business_cost_optimization.png
│       └── 05_confusion_matrices.png
│
├── 5. Global Superstore/
│   ├── app.py                             # Streamlit BI dashboard
│   ├── notebook.ipynb                     # EDA notebook
│   ├── requirements.txt
│   ├── README.md
│   └── dataset/
│       └── Global_Superstore.xls
│
└── README.md                              ← You are here
```

---

## 💼 Key Business Impact

| Business Challenge | How These Projects Help |
|---|---|
| **Improve marketing efficiency** | Task 1 & 2: Score and segment customers before any spend; eliminate wasted outreach |
| **Reduce financial risk** | Task 4: Cost-optimised lending decisions reduce loan loss by ~14% vs naive threshold |
| **Increase customer retention** | Task 2: Segment-specific loyalty strategies prevent churn in high-value clusters |
| **Forecast resource usage** | Task 3: Accurate demand forecasting enables dynamic tariff savings and grid stability |
| **Improve decision-making** | Task 5: Real-time BI dashboard gives management visibility across all KPIs in one place |
| **Optimise profitability** | Task 5: Identifies margin-destroying products and over-discounting behaviour immediately |

---

## 🔮 Future Improvements

| Task | Enhancement | Expected Impact |
|---|---|---|
| **Task 1** | Add XGBoost / LightGBM; remove `duration` for pre-call deployability | +2–4% AUC; truly pre-campaign-usable model |
| **Task 1** | SMOTE oversampling + threshold tuning | Higher recall on the minority subscriber class |
| **Task 2** | Extend to RFM (Recency-Frequency-Monetary) segmentation | Richer behavioural clusters using transaction history |
| **Task 2** | DBSCAN for noise-robust clustering | Handle outlier customers without forcing them into clusters |
| **Task 3** | LSTM / Temporal Fusion Transformer | Better capture of long-range seasonal dependencies |
| **Task 3** | Multi-household aggregation | Scale to grid-level demand forecasting |
| **Task 4** | Add bureau payment history and credit utilisation features | Material AUC improvement; closer to production-grade scoring |
| **Task 4** | Quarterly threshold recalibration pipeline | Keeps cost optimisation aligned with economic conditions |
| **Task 5** | Add ARIMA/Prophet sales forecasting module | Forward-looking revenue projections within the dashboard |
| **Task 5** | Connect to live PostgreSQL / BigQuery source | Real-time KPI updates replacing static file refresh |

---

## 🎯 Conclusion

This repository represents a comprehensive, business-first approach to Data Science across five distinct domains. Every project moves beyond academic metric optimisation to answer the question that actually matters to organisations: *how does this analysis change a business decision and by how much?*

Whether it is identifying which bank customers to call, which shoppers deserve VIP treatment, when household appliances should run to save money on electricity, how conservative a bank should be when approving loans, or which product lines are silently eroding a retailer's margins — each project delivers a quantified, actionable answer grounded in real data and sound methodology.

The breadth of techniques demonstrated — supervised classification, unsupervised clustering, statistical time series modelling, gradient boosted trees, explainable AI, and interactive business intelligence — reflects the full skill set required of a modern Data Scientist or Data Analyst operating at the intersection of technical rigour and business impact.

---

<div align="center">

**Mirza Muhammad Ahsan**
Data Science & Analytics Internship — Phase 2
DevelopersHub Corporation

*Built with Python · scikit-learn · XGBoost · Streamlit · Plotly · Jupyter*

</div>
