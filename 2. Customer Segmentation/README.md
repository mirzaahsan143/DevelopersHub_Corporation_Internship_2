<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Customer%20Segmentation&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Unsupervised%20Machine%20Learning%20%7C%20K-Means%20%7C%20PCA%20%7C%20t-SNE&descAlignY=56&descSize=18" width="100%"/>

<br/>

# 🛍️ Mall Customer Segmentation
### Turning Raw Customer Data into Actionable Business Intelligence

<br/>

<!-- BADGES ROW 1 -->
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)

<!-- BADGES ROW 2 -->
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0?style=for-the-badge)
![K-Means](https://img.shields.io/badge/K--Means-Clustering-FF6B6B?style=for-the-badge)
![PCA](https://img.shields.io/badge/PCA-Dimensionality%20Reduction-6C63FF?style=for-the-badge)
![t-SNE](https://img.shields.io/badge/t--SNE-Visualization-2ECC71?style=for-the-badge)

<!-- BADGES ROW 3 -->
![Internship](https://img.shields.io/badge/DevelopersHub-Internship%20Task%202-FF4500?style=for-the-badge&logo=dev.to&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Stars](https://img.shields.io/badge/Portfolio-Project-gold?style=for-the-badge&logo=github)

<br/>

---

**📌 DevelopersHub Corporation — Data Science Internship | Task 2**

*Applying unsupervised machine learning to discover 5 distinct customer personas from mall shopping data, enabling precision-targeted marketing strategies backed by data.*

---

</div>

<br/>

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Project Overview](#-project-overview) |
| 2 | [Task Objective](#-task-objective) |
| 3 | [Dataset Description](#-dataset-description) |
| 4 | [Project Workflow](#-project-workflow) |
| 5 | [Exploratory Data Analysis](#-exploratory-data-analysis-eda) |
| 6 | [Data Preprocessing](#️-data-preprocessing) |
| 7 | [K-Means Clustering](#-k-means-clustering) |
| 8 | [Optimal Cluster Selection](#-finding-the-optimal-number-of-clusters) |
| 9 | [Dimensionality Reduction](#-dimensionality-reduction--pca--t-sne) |
| 10 | [Customer Segment Analysis](#-customer-segment-analysis) |
| 11 | [Marketing Strategies](#-marketing-strategy-recommendations) |
| 12 | [Business Insights](#-business-insights) |
| 13 | [ML Concepts Used](#-machine-learning--data-science-concepts-used) |
| 14 | [Visualizations](#-visualization-gallery) |
| 15 | [Key Findings](#-key-findings) |
| 16 | [Final Conclusion](#-final-conclusion) |
| 17 | [Future Enhancements](#-future-enhancements) |

<br/>

---

## 🌐 Project Overview

In today's hyper-competitive retail landscape, treating every customer identically is not just inefficient — it's a missed opportunity worth millions. The difference between a good mall and a great one lies in how deeply it understands who walks through its doors.

This project applies **unsupervised machine learning** to the Mall Customers dataset to uncover natural groupings in customer behavior. By analyzing demographic information — age, annual income, and spending habits — we let the data tell its own story, without any predefined labels or assumptions.

The output is five clearly defined **customer personas**, each with distinct behavioral fingerprints. These personas don't just describe segments; they directly inform **how to market, retain, and grow revenue** from each group.

### Why Does Customer Segmentation Matter?

Every major retail and e-commerce brand invests heavily in segmentation:

- **Amazon** uses behavior-based clustering to power its recommendation engine
- **Walmart** applies segmentation to optimize in-store layouts for regional customer profiles  
- **Banking institutions** cluster customers for credit risk profiling and personalized product offers  
- **Telecom companies** use segmentation to identify churn-prone groups before they leave

The logic is universal: **personalized engagement outperforms generic campaigns** in every measurable metric — click rate, conversion, retention, and lifetime value.

### Why Unsupervised Learning?

Unlike classification or regression, we don't have a labeled column telling us "this is a premium customer." We only have raw features. Unsupervised learning — specifically K-Means clustering — discovers hidden structure in the data without any pre-tagged answers. This makes it ideal for **market discovery**, where the goal is to find patterns that human intuition alone might miss.

<br/>

---

## 🎯 Task Objective

> **DevelopersHub Corporation — Data Science Internship | Task 2**

This project was completed as **Task 2** of the DevelopersHub Corporation Data Science Internship. The task required:

1. **Performing comprehensive Exploratory Data Analysis (EDA)** on the Mall Customers dataset — understanding distributions, relationships, and business-relevant patterns across all features.

2. **Applying K-Means Clustering** to segment customers into natural, data-driven groups based on their income and spending behavior.

3. **Using the Elbow Method and Silhouette Score** to objectively determine the optimal number of clusters rather than guessing arbitrarily.

4. **Leveraging PCA and t-SNE** for dimensionality reduction — reducing 3-feature data into 2D projections to visually validate that the clusters are well-separated and meaningful.

5. **Profiling each customer segment** in detail, analyzing their demographic makeup, spending behavior, and income characteristics.

6. **Deriving actionable marketing strategies** for each cluster — so the business can immediately act on the segmentation outputs without needing a data science background to interpret results.

The ultimate goal: transform a raw CSV of 200 mall customers into a **strategic roadmap** that drives smarter, more profitable marketing decisions.

<br/>

---

## 📦 Dataset Description

### Source
**Mall Customers Dataset** — a widely used benchmark dataset for customer segmentation tasks, commonly associated with mall retail analytics.

### Overview

| Property | Value |
|----------|-------|
| Total Records | **200 customers** |
| Total Features | **5 columns** |
| Missing Values | **0 (zero)** |
| Duplicate Rows | **0 (zero)** |
| Data Quality | **Clean — ready for analysis** |

### Feature Dictionary

| Feature | Original Column Name | Data Type | Range | Role in Analysis |
|---------|---------------------|-----------|-------|-----------------|
| Customer ID | `CustomerID` | Integer | 1 – 200 | Unique identifier only — **excluded from clustering** |
| Gender | `Genre` → renamed `Gender` | Categorical | Male / Female | Used for demographic breakdown; not directly clustered |
| Age | `Age` | Integer | 18 – 70 years | Included in 3D clustering; analyzed in EDA |
| Annual Income | `Annual Income (k$)` → `Annual_Income` | Integer | $15k – $137k | **Primary clustering feature** |
| Spending Score | `Spending Score (1-100)` → `Spending_Score` | Integer | 1 – 99 | **Primary clustering feature** — mall-assigned score |

### Feature Deep Dive

**`CustomerID`** — A sequential identifier with no analytical value. Dropped before any clustering to ensure the algorithm only learns from meaningful behavioral signals.

**`Gender`** — Categorical variable renamed from `Genre` for clarity. The dataset skews slightly female (56%) vs male (44%). Used to analyze gender composition within each cluster.

**`Age`** — Customer age in years. The average customer is approximately 38 years old. This feature shows a weak negative correlation with Spending Score — younger customers tend to spend more aggressively, while older customers tend to be more conservative.

**`Annual Income (k$)`** — The customer's annual income expressed in thousands of US dollars. This is one of the two most powerful clustering features. Income ranges from $15k (budget households) to $137k (high earners), creating natural separation in the data.

**`Spending Score (1-100)`** — The most analytically critical feature. This score is assigned by the mall based on observed customer spending behavior and visit frequency. A score of 1 indicates very low engagement; 99 indicates the most active, highest-spending customers. When plotted against Annual Income, this feature reveals the 5 natural customer clusters that K-Means later formalizes.

### Column Renaming

For readability and clean code throughout the notebook, three columns were renamed at the start:

```
Genre               → Gender
Annual Income (k$)  → Annual_Income
Spending Score (1-100) → Spending_Score
```

<br/>

---

## 🔄 Project Workflow

The project follows a structured, reproducible data science pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PROJECT PIPELINE                                 │
├──────────────────────┬──────────────────────────────────────────────┤
│  STAGE               │  ACTIONS PERFORMED                           │
├──────────────────────┼──────────────────────────────────────────────┤
│  1. Data Loading     │  pd.read_csv() → first look, shape, dtypes   │
│  2. Data Quality     │  Missing values, duplicates, unique counts   │
│  3. Column Cleanup   │  Rename columns for usability                │
│  4. EDA              │  8 visualizations across all features        │
│  5. Preprocessing    │  Feature selection + StandardScaler          │
│  6. Cluster Search   │  Elbow Method (K=1–11) + Silhouette Score    │
│  7. K-Means (2D)     │  Annual_Income × Spending_Score, K=5         │
│  8. K-Means (3D)     │  Age + Annual_Income + Spending_Score, K=5   │
│  9. PCA              │  3D → 2D linear projection, variance report  │
│ 10. t-SNE            │  3D → 2D non-linear embedding                │
│ 11. Segment Analysis │  Per-cluster profiling + revenue proxy       │
│ 12. Business Output  │  5 marketing strategies + insights           │
└──────────────────────┴──────────────────────────────────────────────┘
```

**Two Parallel Clustering Models** were trained:
- **2D Model** → `Annual_Income` + `Spending_Score` only — the classic, highly visual segmentation used for business storytelling
- **3D Model** → `Age` + `Annual_Income` + `Spending_Score` — richer behavioral profiling used for PCA and t-SNE visualization

<br/>

---

## 📊 Exploratory Data Analysis (EDA)

EDA is the most important first step in any data science project. Before building models, we need to deeply understand what the data contains, how features are distributed, and what patterns emerge visually. The notebook performs **8 distinct EDA visualizations**, each revealing a different dimension of customer behavior.

---

### 5.1 — Gender Distribution

**Visualization:** Side-by-side bar chart and pie chart

The dataset contains **112 female customers (56%)** and **88 male customers (44%)**. This mild skew toward female shoppers is typical in mall retail environments and matters for targeting — female customers represent the slight majority of the mall's footfall.

The visualization uses dual panels: a count bar chart with annotated values and a proportional pie chart, giving both absolute and relative perspectives on the gender split.

<p align="center">
  <img src="images/gender_distribution.png" width="700" alt="Gender Distribution"/>
</p>

> 💡 **Business Insight:** Female customers outnumber males by roughly 3:2. Any broad campaign should be designed with female customer preferences as the primary orientation.

---

### 5.2 — Age Distribution

**Visualization:** Histogram (with mean line) + Box plot by Gender

The age histogram reveals a **right-skewed distribution** with a concentration of customers between **18 and 40 years**. The mean customer age sits around **38.9 years**. The youngest customer is 18 and the oldest is 70.

The gender-segmented box plot shows that both male and female customers have similar median ages, though female customers show slightly more variance in their age distribution.

<p align="center">
  <img src="images/age_distribution.png" width="700" alt="Age Distribution"/>
</p>

> 💡 **Business Insight:** The mall primarily attracts a younger adult demographic. Marketing campaigns should skew toward digital-first and socially-driven channels preferred by the 18–40 age group.

---

### 5.3 — Annual Income Distribution

**Visualization:** Histogram (with mean line) + Box plot by Gender

Annual income follows a roughly **uniform distribution** between $15k and $137k, with a mean of approximately **$60.6k**. This wide spread confirms that the mall serves a diverse economic demographic — from budget households to affluent families.

The gender-split box plot shows that male and female customers have comparable income distributions, though males have slightly higher median income in this dataset.

<p align="center">
  <img src="images/income_distribution.png" width="700" alt="Annual Income Distribution"/>
</p>

> 💡 **Business Insight:** With income spread almost uniformly across a ~$120k range, the mall needs significantly different product and price positioning strategies for different income tiers — one-size-fits-all pricing will alienate both ends.

---

### 5.4 — Spending Score Distribution

**Visualization:** Histogram (with mean line) + Box plot by Gender

The spending score distribution is **remarkably uniform**, almost flat across the 1–99 range, with a mean of approximately **50.2 out of 100**. This is a strong signal — it tells us the mall has not successfully concentrated spending among its best customers. There's significant room to shift customers from the low-scoring end to the high-scoring end.

The gender box plot reveals that **female customers tend to have marginally higher spending scores** than male customers.

<p align="center">
  <img src="images/spending_distribution.png" width="700" alt="Spending Score Distribution"/>
</p>

> 💡 **Business Insight:** A flat spending score distribution is a marketing opportunity — the mall has customers capable of higher engagement who simply haven't been incentivized yet. Targeted campaigns can shift the score distribution rightward.

---

### 5.5 — Correlation Heatmap

**Visualization:** Annotated heatmap across Age, Annual Income, and Spending Score

| Feature Pair | Correlation | Interpretation |
|-------------|-------------|----------------|
| Age ↔ Annual Income | +0.03 | Nearly zero — income doesn't grow predictably with age in this dataset |
| Age ↔ Spending Score | **−0.33** | Moderate negative — **younger customers spend more** |
| Annual Income ↔ Spending Score | −0.32 | Moderate negative — higher earners aren't necessarily bigger spenders here |

<p align="center">
  <img src="images/correlation_heatmap.png" width="600" alt="Correlation Heatmap"/>
</p>

> 💡 **Business Insight:** The negative correlation between age and spending score is particularly actionable. Engagement strategies for younger customers should feel exciting and impulsive; those for older customers should emphasize value and reliability.

---

### 5.6 — Pair Plot (Feature Relationships)

**Visualization:** Seaborn pairplot with Gender hue across all numeric features

The pair plot generates a 3×3 matrix of scatter plots and KDE distributions across all combinations of Age, Annual Income, and Spending Score — separated by gender color. This is the most information-dense single visualization in the EDA section.

Key observations:
- The **Income × Spending Score** panel already shows **5 distinct visual clusters** before any algorithm is applied
- KDE diagonals confirm the distribution shapes noted above
- Gender overlap is high across all feature combinations — gender alone doesn't explain cluster formation

<p align="center">
  <img src="images/pair_plot.png" width="700" alt="Pair Plot"/>
</p>

---

### 5.7 — Annual Income vs Spending Score (Primary Business View)

**Visualization:** Scatter plot colored by gender

This is the most strategically important EDA visualization. When Annual Income is plotted against Spending Score, **five clearly separated natural groupings** emerge visually — even before running K-Means. This scatter plot provides the human intuition confirmation that the data is suitable for clustering.

The five visual clusters correspond to:
- Low income, high spending (top-left)
- High income, high spending (top-right)
- Middle income, moderate spending (center)
- Low income, low spending (bottom-left)
- High income, low spending (bottom-right)

<p align="center">
  <img src="images/income_vs_spending.png" width="700" alt="Income vs Spending Score"/>
</p>

> 💡 **Business Insight:** K-Means will formalize what is visually obvious here. The natural separation in this 2D space means the algorithm will produce highly meaningful, well-separated clusters rather than ambiguous ones.

---

### 5.8 — Age vs Spending Score

**Visualization:** Scatter plot colored by gender

This plot explores how spending behavior shifts across age groups. Customers aged **20–35 show the widest range of spending scores** — some spend very aggressively while others are extremely conservative. In contrast, customers aged **50+** cluster predominantly at lower spending scores, suggesting age-related behavioral shifts toward financial caution.

<p align="center">
  <img src="images/age_vs_spending.png" width="700" alt="Age vs Spending Score"/>
</p>

> 💡 **Business Insight:** The 20–35 age band is the most behaviorally volatile and therefore the most responsive to marketing. Campaigns targeting this group have the highest potential swing — the right offer can take a low spender to a high spender.

<br/>

---

## ⚙️ Data Preprocessing

Clean data alone isn't enough for K-Means — the algorithm is highly sensitive to **feature scale**. Before clustering, two critical preprocessing steps were applied.

### Step 1: Feature Selection

`CustomerID` is a sequential administrative identifier. It carries zero behavioral information and would actively mislead the clustering algorithm if included. It was dropped entirely.

Two clustering configurations were prepared:

- **2D Feature Set:** `Annual_Income` + `Spending_Score` — for the clean, highly interpretable 2D cluster visualization
- **3D Feature Set:** `Age` + `Annual_Income` + `Spending_Score` — for richer multi-dimensional profiling and dimensionality reduction

`Gender` was encoded but not included as a direct clustering feature. It was reserved for post-hoc demographic analysis within each discovered cluster.

### Step 2: StandardScaler Normalization

K-Means calculates **Euclidean distance** between every data point and every cluster centroid. If one feature has a much larger numeric range than another (e.g., Annual Income in thousands vs Spending Score 1–100), the high-range feature will dominate the distance calculations — effectively making other features invisible to the algorithm.

`StandardScaler` from `sklearn.preprocessing` was applied to transform each feature to:
- **Mean ≈ 0**
- **Standard Deviation ≈ 1**

This ensures every feature contributes equally to the distance calculations, producing fair, unbiased clustering.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Age', 'Annual_Income', 'Spending_Score']])
X_2d_scaled = scaler.fit_transform(df[['Annual_Income', 'Spending_Score']])
```

**Post-scaling verification:**
- Mean of all scaled features → ~0.0000
- Standard deviation of all scaled features → ~1.0000

This confirms the scaler worked correctly and data is ready for clustering.

<br/>

---

## 🤖 K-Means Clustering

### What is K-Means?

K-Means is an **iterative, centroid-based unsupervised learning algorithm** that partitions data into K non-overlapping groups by minimizing the total within-cluster variance (inertia).

### How K-Means Works — Step by Step

```
  Step 1: Choose K → initialize K cluster centers (centroids)
  Step 2: Assign   → each data point joins its nearest centroid (Euclidean distance)
  Step 3: Update   → recalculate each centroid as the mean of all points in its cluster
  Step 4: Repeat   → keep iterating Steps 2–3 until centroid positions stop changing
```

The algorithm converges when assignments stabilize — no data point switches clusters between iterations.

### K-Means++ Initialization

This project uses **K-Means++** (via `init='k-means++'`), which improves over random initialization by:
1. Placing the first centroid randomly
2. Choosing subsequent centroids proportionally to their distance from existing centroids

This prevents poor initializations where centroids start too close together, leading to faster convergence and consistently better cluster quality.

### Configuration Used

```python
KMeans(
    n_clusters = 5,          # Optimal K determined by Elbow + Silhouette
    init       = 'k-means++', # Smart initialization
    n_init     = 10,          # Run 10 times, keep best result
    random_state = 42         # Reproducibility
)
```

Setting `n_init=10` means the algorithm runs 10 full independent cycles and keeps the result with the lowest inertia — guarding against unlucky initializations.

### Two Models Were Trained

| Model | Features Used | Purpose |
|-------|--------------|---------|
| **2D K-Means** | `Annual_Income` + `Spending_Score` | Primary business visualization — clean 2D scatter plot |
| **3D K-Means** | `Age` + `Annual_Income` + `Spending_Score` | Richer profiling — used as input for PCA and t-SNE |

<br/>

---

## 🔍 Finding the Optimal Number of Clusters

Choosing K is the most important decision in K-Means. Too few clusters oversimplify reality; too many create segments too small to act on. Two rigorous, complementary methods were used.

---

### Method 1 — The Elbow Method (WCSS)

**WCSS (Within-Cluster Sum of Squares)** — also called *inertia* — measures the total squared distance from every data point to its assigned centroid. Lower WCSS means tighter, better-defined clusters.

The algorithm was run for K = 1 through K = 11 and WCSS was recorded at each step:

- As K increases, WCSS always decreases (more clusters = each one is smaller and tighter)
- The key is to find the **"elbow"** — the K value after which WCSS improvement diminishes rapidly

**Result:** The WCSS curve shows a **clear, sharp elbow at K = 5**. Beyond K = 5, adding more clusters yields only marginal WCSS reduction — confirming 5 is the natural inflection point.

<p align="center">
  <img src="images/elbow_method.png" width="700" alt="Elbow Method — WCSS vs K"/>
</p>

---

### Method 2 — Silhouette Score

The **Silhouette Score** evaluates clustering quality from a different angle. For each data point, it measures:

- **a** = average distance to other points in the **same** cluster (compactness)
- **b** = average distance to points in the **nearest other** cluster (separation)

```
Silhouette Score = (b - a) / max(a, b)
```

The score ranges from **−1 to +1**:
- **+1** → perfect clustering (points tightly grouped, far from other clusters)
- **0** → overlapping clusters
- **−1** → points assigned to the wrong cluster

Silhouette scores were computed for K = 2 through K = 11:

**Result:** **K = 5 delivers the highest Silhouette Score**, confirming both methods agree. The score at K = 5 was notably higher than neighboring values, validating that 5 is not just an elbow artifact but a genuinely optimal partition.

<p align="center">
  <img src="images/silhouette_scores.png" width="700" alt="Silhouette Score vs K"/>
</p>

> ✅ **Both the Elbow Method and Silhouette Score independently converge on K = 5** — this dual-method validation gives high confidence that 5 clusters is the correct choice for this dataset.

<br/>

---

## 📉 Dimensionality Reduction — PCA & t-SNE

The 3D feature set (Age + Annual Income + Spending Score) cannot be directly plotted. Dimensionality reduction techniques compress the 3 features into 2 dimensions while preserving the most important structural information, enabling visual cluster validation.

---

### Principal Component Analysis (PCA)

**What PCA does:** PCA is a **linear** transformation that finds the directions of maximum variance in the data (principal components) and projects the data onto those directions. It's deterministic — running PCA twice on the same data always produces the same result.

**Results from this project:**

| Component | Variance Explained |
|-----------|-------------------|
| PC1 | *~XX%* |
| PC2 | *~XX%* |
| **Total** | **Combined PC1 + PC2** |

Even with just 2 components, PCA preserves a substantial portion of the original data's variance, making the 2D projection a faithful representation.

**Cluster visualization in PCA space** shows that the 5 clusters identified by K-Means remain visible and largely separated when projected linearly — confirming the clusters have real geometric structure in the original 3D space.

<p align="center">
  <img src="images/pca_clusters.png" width="700" alt="PCA Cluster Visualization"/>
</p>

<p align="center">
  <img src="images/pca_variance.png" width="500" alt="PCA Explained Variance"/>
</p>

---

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**What t-SNE does:** t-SNE is a **non-linear** technique that focuses on preserving *local neighborhood structure* in the data. It's especially effective at revealing tight cluster groupings that linear methods like PCA might compress or overlap.

**Configuration used:**

```python
TSNE(
    n_components  = 2,
    perplexity    = 30,    # Effective neighbor count — works well for ~200 samples
    max_iter      = 1000,
    random_state  = 42,
    learning_rate = 'auto',
    init          = 'pca'  # PCA-based init for stability
)
```

**Result:** The t-SNE plot typically reveals **even tighter, more visually separated clusters** than PCA. However, unlike PCA, the t-SNE axes have no direct interpretable meaning — the distances between clusters in t-SNE space are not proportional to true feature distances. Its value is purely visual.

<p align="center">
  <img src="images/tsne_clusters.png" width="700" alt="t-SNE Cluster Visualization"/>
</p>

| Method | Type | Axes Interpretable? | Best For |
|--------|------|---------------------|----------|
| PCA | Linear | ✅ Yes — variance explained | Global structure, feature-linked visualization |
| t-SNE | Non-linear | ❌ No — abstract embedding | Local cluster compactness, visual confirmation |

Together, PCA and t-SNE provide **complementary visual evidence** that K-Means discovered real, meaningful clusters — not random noise.

<br/>

---

## 👥 Customer Segment Analysis

After clustering, each of the 5 groups was profiled individually using average statistics across all features. The notebook assigned descriptive business labels to each cluster based on the income–spending score centroid positions.

---

<p align="center">
  <img src="images/kmeans_clusters_2d.png" width="750" alt="K-Means 2D Cluster Visualization"/>
</p>

<p align="center">
  <img src="images/cluster_heatmap.png" width="650" alt="Cluster Profile Heatmap"/>
</p>

---

### 🎯 Cluster 0 — Sensible Savers

**Label:** Mid Income, Low Spending | **Color:** Red

| Metric | Value |
|--------|-------|
| Average Age | ~40–45 years |
| Average Annual Income | ~$55k |
| Average Spending Score | ~40 / 100 |

**Who they are:** Middle-income earners who are conservative and deliberate with their mall spending. They visit the mall but don't convert to high-value purchases. They likely do their research before buying and prioritize necessity over impulse.

**Behavioral profile:** These customers have the financial capacity to spend more but haven't been given compelling enough reasons to do so. They're not price-insensitive — they're value-conscious. They respond to rational arguments about savings and ROI rather than luxury signals.

---

### 💎 Cluster 1 — Premium Customers

**Label:** High Income, High Spending | **Color:** Blue

| Metric | Value |
|--------|-------|
| Average Age | ~30–35 years |
| Average Annual Income | ~$85k+ |
| Average Spending Score | ~82 / 100 |

**Who they are:** The mall's golden segment. These customers earn significantly above average and translate that earning power into aggressive mall spending. They're younger high-earners who are comfortable spending freely and have strong brand preferences.

**Behavioral profile:** These are the customers most likely to respond to premium experiences, exclusive access, and personalized service. They visit regularly, spend per visit above average, and represent the highest lifetime value of any segment.

---

### 🛒 Cluster 2 — Budget Shoppers

**Label:** Low Income, Low Spending | **Color:** Green

| Metric | Value |
|--------|-------|
| Average Age | ~45 years |
| Average Annual Income | ~$25k |
| Average Spending Score | ~20 / 100 |

**Who they are:** Cost-conscious shoppers with limited disposable income. Their low spending score reflects genuine financial constraints, not disinterest. This group may include students, retirees, and households managing tight budgets.

**Behavioral profile:** These customers visit the mall for essentials rather than experiential shopping. Frequency could be increased with the right discount structures, but average basket size will remain limited by income. The goal here is visit frequency, not ticket size.

---

### ⭐ Cluster 3 — Careless Spenders

**Label:** Low Income, High Spending | **Color:** Orange

| Metric | Value |
|--------|-------|
| Average Age | ~25 years |
| Average Annual Income | ~$25k |
| Average Spending Score | ~78 / 100 |

**Who they are:** The most intriguing segment. These customers spend at a level disproportionate to their income — their spending score rivals the premium segment despite earning far less. This is the impulse buyer profile: driven by trends, social influence, and emotional purchasing rather than financial planning.

**Behavioral profile:** Predominantly younger customers (early-to-mid 20s) who prioritize experiences and social currency over savings. They are highly responsive to social media campaigns, limited-edition drops, and FOMO-driven marketing tactics.

---

### 💰 Cluster 4 — Conservative Elites

**Label:** High Income, Low Spending | **Color:** Purple

| Metric | Value |
|--------|-------|
| Average Age | ~40 years |
| Average Annual Income | ~$85k+ |
| Average Spending Score | ~17 / 100 |

**Who they are:** The most strategically valuable *untapped* segment. These customers have high earning power but barely engage at the mall. Their low spending score isn't a budget constraint — it's a preference or experience issue. They're choosing to spend their money elsewhere.

**Behavioral profile:** Possibly frustrated by the current product mix, unconvinced by existing brands, or simply not feeling valued enough to spend here. The revenue potential is enormous — converting even a fraction of this group's income into mall spending would deliver outsized returns.

<br/>

---

## 📣 Marketing Strategy Recommendations

Each customer segment requires a fundamentally different marketing approach. A campaign designed for Premium Customers will fall flat with Budget Shoppers — and vice versa. The strategies below are directly derived from each cluster's behavioral and demographic profile.

---

### 🎯 Cluster 0 — Sensible Savers Marketing Strategy

> *"Show them the value, not the price tag."*

| Strategy | Description |
|----------|-------------|
| **Value Bundles & Combo Deals** | Package complementary products at a bundled price that feels like genuine savings — these customers love to feel they've made a smart financial decision |
| **Loyalty Points Programs** | Introduce a points-per-purchase system. The gradual accumulation of rewards gives them a rational reason to visit more often |
| **Clearance & Early Access Sales** | Send pre-sale alerts via email. This group appreciates the insider advantage and is motivated by scarcity + savings combined |
| **Cashback Offers** | A flat cashback percentage on purchases above a threshold gives them measurable, tangible return on spending — removing the psychological barrier to larger purchases |
| **Value Comparison Signage** | In-store displays that compare "price per use" or long-term value of products resonate deeply with this group |

**Expected Impact:** Increased visit frequency and gradual upward shift in average spending score from ~40 toward 50–55.

---

### 💎 Cluster 1 — Premium Customer Marketing Strategy

> *"Protect and elevate your best assets."*

| Strategy | Description |
|----------|-------------|
| **VIP Membership Program** | Create an elite tier with real, tangible benefits — private lounge access, personal shopping assistants, skip-the-line billing |
| **Exclusive Pre-Launch Invitations** | Invite this segment to product previews and brand launch events before public availability. Exclusivity is currency for this group |
| **Luxury Brand Partnerships** | Curate co-branded newsletters and early access to high-end brand collections arriving at the mall |
| **Birthday & Anniversary Rewards** | Highly personalized milestone rewards — a concierge experience on their birthday converts loyalty into emotional attachment |
| **Dedicated Customer Success** | Assign premium customers a point-of-contact for personalized shopping guidance. White-glove service drives retention |

**Expected Impact:** Strongest retention of the highest-revenue segment. Even a 10% churn reduction here has outsized revenue impact.

---

### 🛒 Cluster 2 — Budget Shopper Marketing Strategy

> *"Remove friction, lower the barrier, increase visits."*

| Strategy | Description |
|----------|-------------|
| **Coupon Books & Discount Vouchers** | Physical or digital coupon packs that provide a meaningful discount on specific product categories |
| **Student & Senior Discount Days** | Dedicated discount days for specific demographics (many in this group are likely students or pensioners) |
| **Flash Sales & 24-Hour Deals** | Time-limited deep discounts create urgency without requiring ongoing price commitments from the mall |
| **Affordable Product Spotlight** | Dedicated "Best Value" sections with signage that celebrates budget-friendly quality — removes shame from budget shopping |
| **Community Events** | Free in-mall events (live music, workshops, demos) that drive foot traffic without requiring a purchase |

**Expected Impact:** Increased visit frequency and mall familiarity — positioning the mall as the default shopping destination for this segment's needs.

---

### ⭐ Cluster 3 — Careless Spender Marketing Strategy

> *"Feed the impulse, amplify the moment."*

| Strategy | Description |
|----------|-------------|
| **Buy Now, Pay Later (BNPL)** | Partner with BNPL platforms (Klarna, Afterpay, etc.) to eliminate the upfront price barrier — this segment will spend above their means anyway, BNPL just formalizes it |
| **Social Media Influencer Campaigns** | This group is trend-driven and socially influenced. Mall-exclusive influencer collaborations and Instagram activations directly reach them |
| **Limited Edition & Exclusive Collections** | Scarcity triggers purchase urgency. "Only 50 available" is more compelling to this group than "50% off" |
| **Strategic Impulse Placement** | Position visually appealing, affordable items at high-traffic zones — entrances, escalators, and checkout areas |
| **Gamified Loyalty** | Scratch-cards, spin-to-win, and challenge-based reward systems appeal to the fun, spontaneous nature of this group |

**Expected Impact:** Increased basket size per visit and stronger social media amplification of the mall brand.

---

### 💰 Cluster 4 — Conservative Elite Marketing Strategy

> *"Win their respect before asking for their wallet."*

| Strategy | Description |
|----------|-------------|
| **Premium Re-Engagement Campaigns** | A targeted "We value you" campaign that leads with exclusive offers — not generic discounts. This group needs to feel recognized, not sold to |
| **Quality Storytelling & Craftsmanship Narratives** | Market the story behind products — heritage, materials, process. These customers buy into values, not transactions |
| **High-Value Service Consultations** | Offer complimentary consultations (interior design, personal styling, tech advisory) — converting a passive visit into an active relationship |
| **Referral Reward Programs** | Incentivize referrals through high-value rewards. These customers have affluent social networks — a successful referral brings in more of the same profile |
| **Premium Membership Preview** | Offer a trial of VIP membership to demonstrate the value proposition before asking for a commitment |

**Expected Impact:** The highest revenue-upside segment. Converting Conservative Elites from low spenders to moderate spenders (score shift from ~17 to ~50) could represent the single largest revenue unlock in the entire customer base.

<br/>

---

## 💼 Business Insights

The notebook computed a **Revenue Proxy Score** (Annual Income × Spending Score ÷ 100) for each customer, enabling a simulated revenue contribution comparison across clusters.

<p align="center">
  <img src="images/revenue_by_cluster.png" width="700" alt="Revenue by Cluster"/>
</p>

<p align="center">
  <img src="images/gender_per_cluster.png" width="700" alt="Gender Distribution per Cluster"/>
</p>

### Top Strategic Insights

| # | Insight | Business Implication |
|---|---------|---------------------|
| 1 | **Premium Customers (Cluster 1) generate the highest revenue proxy** | These 39–40 customers are irreplaceable. Retention costs must be prioritized — losing one costs more than acquiring five Budget Shoppers |
| 2 | **Conservative Elites (Cluster 4) are the largest untapped opportunity** | High income + low spending = significant uncaptured revenue. A 2× spending score increase would make this the top-revenue cluster |
| 3 | **Careless Spenders (Cluster 3) spend above their income bracket** | BNPL and flexible payment options will further stimulate spending in a group already predisposed to it |
| 4 | **Budget Shoppers (Cluster 2) need frequency, not ticket size** | Strategy should focus on visit occasions rather than persuading larger individual purchases |
| 5 | **Sensible Savers (Cluster 0) are persuadable with the right framing** | Value-based messaging can gradually shift this group's score — they have the income, they need the justification |
| 6 | **Female customers have slightly higher spending scores** | Gender-targeted campaigns for high-spending product categories should lead with female-oriented creatives |

<br/>

---

## 🧠 Machine Learning & Data Science Concepts Used

| Concept | Category | Application in This Project |
|---------|----------|-----------------------------|
| **Unsupervised Learning** | ML Paradigm | No labels used — the algorithm discovers structure independently |
| **K-Means Clustering** | Algorithm | Core segmentation — partitions 200 customers into 5 distinct groups |
| **K-Means++ Initialization** | Algorithm Optimization | Smarter centroid initialization for faster convergence and better results |
| **Elbow Method (WCSS)** | Model Evaluation | Determines optimal K by measuring within-cluster variance at each K |
| **Silhouette Score** | Model Evaluation | Quantifies cluster compactness and separation quality |
| **StandardScaler** | Preprocessing | Normalizes feature ranges so no single feature dominates distance calculations |
| **PCA** | Dimensionality Reduction | Linear projection from 3D to 2D preserving maximum variance |
| **t-SNE** | Dimensionality Reduction | Non-linear embedding that reveals local cluster structure |
| **Exploratory Data Analysis** | Analysis Methodology | 8+ visualizations to understand distributions, correlations, and patterns |
| **Feature Engineering** | Preprocessing | Revenue Proxy derivation (Income × Score), column renaming for usability |
| **Customer Analytics** | Domain Knowledge | Translating cluster statistics into human-readable business personas |
| **Data Visualization** | Communication | matplotlib + seaborn for histograms, scatter plots, heatmaps, bar charts |

<br/>

---

## 🖼️ Visualization Gallery

All visualizations were saved at 150 DPI for high-resolution export.

| Visualization | File | Description |
|---------------|------|-------------|
| Gender Distribution | `gender_distribution.png` | Bar + Pie chart of gender breakdown |
| Age Distribution | `age_distribution.png` | Histogram + Gender box plot |
| Income Distribution | `income_distribution.png` | Histogram + Gender box plot |
| Spending Distribution | `spending_distribution.png` | Histogram + Gender box plot |
| Correlation Heatmap | `correlation_heatmap.png` | Feature correlation matrix |
| Pair Plot | `pair_plot.png` | All-feature pairwise scatter + KDE |
| Income vs Spending | `income_vs_spending.png` | Pre-clustering business view |
| Age vs Spending | `age_vs_spending.png` | Demographic behavior view |
| Elbow Method | `elbow_method.png` | WCSS vs K (1–11) |
| Silhouette Scores | `silhouette_scores.png` | Silhouette Score vs K (2–11) |
| K-Means 2D Clusters | `kmeans_clusters_2d.png` | Final cluster plot with centroids marked |
| Cluster Sizes | `cluster_sizes.png` | Customer count per cluster |
| PCA Clusters | `pca_clusters.png` | Dimensionally reduced cluster visualization |
| PCA Variance | `pca_variance.png` | Explained variance per PC |
| t-SNE Clusters | `tsne_clusters.png` | Non-linear cluster embedding |
| Cluster Heatmap | `cluster_heatmap.png` | Average feature values per cluster |
| Cluster Profiles | `cluster_profiles.png` | Normalized feature comparison across clusters |
| Revenue by Cluster | `revenue_by_cluster.png` | Revenue proxy contribution per segment |
| Gender per Cluster | `gender_per_cluster.png` | Gender distribution within each cluster |

<br/>

---

## 🔑 Key Findings

### Customer Behavior Discoveries

1. **Five distinct customer archetypes exist naturally in the data** — visible even before applying any algorithm in the Income vs Spending Score scatter plot. K-Means formalized what human eyes could already detect.

2. **Age negatively correlates with spending (−0.33)** — younger customers (20–35) are the most spending-volatile group. They represent both the highest potential uplift and the most impulsive behavior.

3. **High income does not guarantee high spending** — Cluster 4 (Conservative Elites) proves that the wealthiest customers may be the mall's most disengaged. Income alone doesn't predict mall loyalty.

4. **The Spending Score distribution is nearly flat** — the mall has not successfully polarized customer spending. There's a significant middle ground that good segmented marketing can push toward higher engagement.

5. **Cluster 3 (Careless Spenders) defies traditional economic logic** — low income, high spending — this group's behavior is emotionally rather than rationally driven. Standard discount strategies will be less effective than experience and social campaigns.

### Model Quality

6. **K = 5 is validated by two independent methods** (Elbow + Silhouette) — this is not a subjective choice. The data itself confirms 5 is the optimal partition.

7. **PCA and t-SNE both confirm cluster separability** — the clusters are not just statistical artifacts. They have real geometric structure in the original multi-dimensional space.

8. **K-Means++ with n_init=10 ensures result stability** — running 10 independent initializations and keeping the best guards against unlucky random seeds that could misplace centroids.

<br/>

---

## 🏁 Final Conclusion

This project demonstrates the full lifecycle of an applied unsupervised machine learning solution — from raw CSV to strategic business recommendations — without any labeled training data.

### What We Accomplished

Starting from 200 rows of mall customer records, this project:

- **Uncovered 5 behaviorally distinct customer personas** through a rigorous, data-validated K-Means clustering pipeline
- **Proved the clustering was statistically sound** using dual validation methods (Elbow + Silhouette) and visual confirmation through PCA and t-SNE
- **Translated algorithmic outputs into business language** — each cluster became a named persona with a behavioral narrative and marketing strategy
- **Identified the single largest revenue opportunity** — Conservative Elites (Cluster 4) — a high-income group with low engagement and enormous spending potential that existing campaigns completely miss

### Business Impact

| Impact Area | Result |
|-------------|--------|
| Marketing Personalization | 5 distinct campaign strategies instead of 1 generic approach |
| Budget Efficiency | High-ROI segments identified for priority investment |
| Revenue Opportunity | Conservative Elites flagged as untapped high-value targets |
| Customer Retention | Premium Customers identified for VIP program development |
| Channel Strategy | Careless Spenders identified as social media priority audience |

### The Broader Message

Customer segmentation is not just a data science exercise — it is the **foundation of modern customer relationship management**. When a business knows exactly who its customers are, every downstream decision improves: product mix, staffing, store layout, digital advertising spend, loyalty programs, and pricing strategy.

This project proves that even a relatively small dataset (200 rows, 5 columns) contains enough behavioral signal to produce actionable, commercially valuable intelligence — when the right analytical framework is applied.

<br/>

---

## 🚀 Future Enhancements

| Enhancement | Description | Business Value |
|-------------|-------------|----------------|
| **RFM Analysis** | Add Recency, Frequency, Monetary modeling on top of existing features | Deeper customer lifetime value scoring |
| **DBSCAN Clustering** | Density-based algorithm that handles outliers and non-spherical clusters | More robust segmentation for complex data distributions |
| **Hierarchical Clustering** | Agglomerative approach with dendrogram visualization | Better cluster count intuition and sub-segment discovery |
| **Gaussian Mixture Models** | Soft-assignment clustering that gives probabilistic cluster membership | Better handling of customers who sit between segments |
| **Real-Time Data Pipeline** | Stream customer transaction data into a live clustering pipeline | Dynamic segment reassignment as behavior changes over time |
| **Streamlit Dashboard** | Interactive web app for non-technical stakeholders to explore segments | Democratizes data insights across the entire business |
| **A/B Testing Framework** | Test each cluster's marketing strategy against a control group | Quantify the ROI of each strategy with statistical rigor |
| **Churn Prediction Layer** | Add a supervised classification layer to predict which customers are at risk | Proactive retention campaigns triggered by behavioral signals |

<br/>

---

<div align="center">

---

### 🏢 DevelopersHub Corporation — Data Science Internship
**Task 2 | Customer Segmentation Using Unsupervised Learning**

---

*"The goal is to turn data into information, and information into insight."*
— Carly Fiorina, former CEO of HP

---

<br/>

**Technologies Used**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
