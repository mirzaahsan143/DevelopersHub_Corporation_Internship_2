# 🛍️ Mall Customer Segmentation Engine
### Demographic Behavioral Analytics & Unsupervised Machine Learning Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Data%20Science-Customer%20Analytics-blue?style=for-the-badge&logo=python&logoColor=white" alt="Data Science">
  <img src="https://img.shields.io/badge/ML-Unsupervised%20Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Clustering-K--Means%20%7C%20K%3D5-green?style=for-the-badge" alt="K-Means">
  <img src="https://img.shields.io/badge/Manifold%20Learning-PCA%20%7C%20t--SNE-red?style=for-the-badge" alt="PCA & t-SNE">
  <img src="https://img.shields.io/badge/DevelopersHub-Internship%20Task%202-blueviolet?style=for-the-badge&logo=dev.to&logoColor=white" alt="Internship">
</p>

---

## 📌 Project Overview

In modern retail analytics, understanding the behavioral fingerprints of distinct customer demographics is essential for maximizing marketing ROI and sustaining revenue growth. This project implements a complete **unsupervised machine learning pipeline** centered on **K-Means Clustering** to segment mall customers based on their demographic profile and observed spending behavior.

Unsupervised learning is the correct paradigm here because no labeled target variable exists — the data contains no pre-assigned "customer type" column. The algorithm is given only raw feature vectors and tasked with discovering hidden structure independently. The result is a set of natural, data-validated customer archetypes that directly inform how the business allocates marketing budget, designs loyalty programs, and prioritizes retention investment.

### Real-World Business Value

* **Precision-Targeted Campaigns:** Segment-specific campaigns consistently outperform generic broadcasts across every measurable metric — click-through rate, conversion, average order value, and long-term retention. This pipeline provides the segmentation infrastructure that makes precision targeting possible.
* **Revenue Uplift from Untapped Segments:** The analysis identifies high-income customer groups with disproportionately low spending — a major, structurally overlooked revenue opportunity that blanket marketing strategies fail to capture.
* **Budget Efficiency:** By concentrating retention spend on the highest-revenue segments and using low-cost automated campaigns for low-engagement groups, the business maximizes return on every marketing dollar.

---

## 🎯 Task & Internship Objectives

> **DevelopersHub Corporation — Data Science Internship | Task 2**

The primary business and technical goal of this analysis is to transform a raw demographic and behavioral dataset into operational customer intelligence.

* **Comprehensive Exploratory Data Analysis (EDA):** Systematically investigate all feature distributions, inter-feature correlations, and gender-segmented behavioral patterns before any modeling is applied.
* **Data Preprocessing & Normalization:** Select analytically meaningful features, discard identifier noise, and apply standard scaling to prepare the feature matrix for distance-based clustering.
* **Optimal Cluster Count Validation:** Apply two independent statistical methods — the Elbow Method (WCSS) and the Silhouette Coefficient — to objectively determine the correct value of K rather than selecting it arbitrarily.
* **K-Means Clustering:** Train an optimized K-Means model on both 2D (Income × Spending Score) and 3D (Age + Income + Spending Score) feature configurations to produce customer segments with distinct behavioral profiles.
* **Dimensionality Reduction:** Apply PCA and t-SNE to project the multi-dimensional cluster space into interpretable 2D visualizations, validating geometric cluster separability.
* **Strategic Business Recommendations:** Translate each statistical cluster into a named customer archetype with a concrete, actionable marketing strategy.

---

## 📊 Dataset Description & Data Schema

The analysis utilizes the **Mall Customers Dataset**, a structured demographic and behavioral record of 200 mall visitors commonly referenced in retail customer analytics research.

* **Total Records:** 200 customers
* **Total Features:** 5 columns
* **Missing Values:** 0 (zero — the dataset is fully populated)
* **Duplicate Rows:** 0 (zero — all records are unique)
* **Data Quality Status:** Clean and analysis-ready with no imputation required

### Features & Structural Typology

| Variable Name | Original Column Name | Structural Type | Value Range | Functional Domain | Technical Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`CustomerID`** | `CustomerID` | Integer | 1 – 200 | Administrative Identifier | Sequential unique ID per customer account. Carries no behavioral signal — **excluded from all clustering operations.** |
| **`Gender`** | `Genre` | Categorical | Male / Female | Demographic Feature | Binary categorical variable renamed from `Genre` for code clarity. Used for post-hoc demographic profiling within discovered clusters. |
| **`Age`** | `Age` | Integer | 18 – 70 years | Demographic Feature | Customer age in years. Included in 3D clustering and examined for behavioral correlation with spending patterns. |
| **`Annual_Income`** | `Annual Income (k$)` | Integer | $15k – $137k | Financial Variable | Annual customer income expressed in thousands of USD. One of two primary clustering features driving segment separation. |
| **`Spending_Score`** | `Spending Score (1-100)` | Integer | 1 – 99 | Behavioral Index | Mall-assigned behavioral score reflecting purchase volume, visit frequency, and overall engagement. The most analytically critical feature in the dataset. |

### Column Renaming Protocol

Three columns were renamed at preprocessing initialization for clean, consistent code access throughout the pipeline:

```
Genre                    →  Gender
Annual Income (k$)       →  Annual_Income
Spending Score (1-100)   →  Spending_Score
```

---

## 🔍 Data Quality Audit & Cleaning Metrics

Although this dataset required no imputation, a full quality audit was executed before analysis to confirm structural integrity and document baseline statistics.

1. **Missing Value Scan:**
   * **Outcome:** Zero missing values across all 5 columns and 200 rows. No imputation strategy required.

2. **Duplicate Row Detection:**
   * **Outcome:** Zero duplicate records identified. All 200 rows represent unique customer observations.

3. **Identifier Column Removal:**
   * **Issue:** `CustomerID` is a sequential administrative key with no analytical value. If retained, it would introduce meaningless numeric variance that could distort cluster centroid calculations.
   * **Action:** Dropped from the clustering feature matrix before model training. Retained in the original DataFrame for traceability only.

4. **Categorical Variable Audit:**
   * `Gender` contains exactly two unique values: `Male` and `Female`. No misspellings, encoding errors, or unexpected categories were present.
   * Gender distribution: **112 Female (56.0%) / 88 Male (44.0%).**

5. **Numerical Range Validation:**
   * All numerical features fall within expected real-world ranges. No negative values in `Age`, `Annual_Income`, or `Spending_Score`. No zero-value anomalies requiring filtering.

---

## 📈 Exploratory Data Analysis (EDA) & Core Findings

Exploratory Data Analysis was conducted across 8 distinct visualizations covering all feature dimensions before any model was trained. The EDA phase serves a dual function: validating data integrity and providing pre-model intuition about cluster structure.

### 1. Gender Distribution Analysis

The dataset skews 56% female and 44% male — a distribution typical of mall retail environments where female shoppers represent the primary footfall demographic. Both a count bar chart and proportional pie chart were generated for absolute and relative perspectives.

> 💡 **Finding:** Female customers outnumber males by approximately 3:2. Marketing creatives targeting broad audiences should be calibrated with female customer preferences as the primary orientation.

<p align="center">
  <img src="images/gender_distribution.png" width="700" alt="Gender Distribution — Count and Proportion"/>
</p>

---

### 2. Age Distribution Analysis

The age histogram reveals a **right-skewed distribution** concentrated between 18 and 40 years, with a mean of approximately **38.9 years**. The distribution spans from age 18 (minimum) to age 70 (maximum). Gender-segmented box plots show comparable median ages for male and female customers with slightly greater variance in the female cohort.

> 💡 **Finding:** The primary customer demographic is younger adults. The 20–35 band represents the highest footfall concentration — a group that responds to digital-first, socially driven marketing channels.

<p align="center">
  <img src="images/age_distribution.png" width="700" alt="Age Distribution — Histogram and Gender Box Plot"/>
</p>

---

### 3. Annual Income Distribution Analysis

Annual income follows a near-uniform spread between $15k and $137k, with a mean of approximately **$60.6k**. The wide income range — spanning over $120k — confirms the mall serves customers across the full economic spectrum from budget households to high-net-worth individuals. Gender-segmented box plots show broadly comparable income distributions between male and female customers.

> 💡 **Finding:** Income dispersion across a $120k range makes a single pricing and product strategy structurally inadequate. Income-tier-specific positioning is essential for capturing spending from both ends of the distribution.

<p align="center">
  <img src="images/income_distribution.png" width="700" alt="Annual Income Distribution — Histogram and Gender Box Plot"/>
</p>

---

### 4. Spending Score Distribution Analysis

The spending score distribution is remarkably flat, with scores spread near-uniformly across the 1–99 range and a mean of approximately **50.2 / 100**. This uniform distribution is analytically significant: it indicates the mall has not successfully concentrated high-engagement behavior among a loyal premium cohort. Gender box plots reveal that **female customers carry marginally higher median spending scores** than male customers.

> 💡 **Finding:** A flat spending score distribution signals a structural marketing gap. Significant portions of the customer base with the capacity to score higher have not been incentivized to do so — representing a direct revenue uplift opportunity.

<p align="center">
  <img src="images/spending_distribution.png" width="700" alt="Spending Score Distribution — Histogram and Gender Box Plot"/>
</p>

---

### 5. Correlation Heatmap — Feature Interdependencies

Pearson correlation coefficients were computed across all three numerical features:

| Feature Pair | Correlation Coefficient | Interpretation |
| :--- | :---: | :--- |
| Age ↔ Annual Income | **+0.03** | Near-zero — income does not scale predictably with age in this dataset |
| Age ↔ Spending Score | **−0.33** | Moderate negative — younger customers exhibit significantly higher spending |
| Annual Income ↔ Spending Score | **−0.32** | Moderate negative — high earners are not proportionally higher spenders |

> 💡 **Finding:** The negative correlation between income and spending score is the most strategically important result in the EDA. High-income customers who are low spenders represent the clearest opportunity for targeted re-engagement — they have the financial capacity that spending score does not currently reflect.

<p align="center">
  <img src="images/correlation_heatmap.png" width="600" alt="Correlation Heatmap — Numerical Features"/>
</p>

---

### 6. Pairwise Feature Relationship Matrix (Pair Plot)

A full seaborn pairplot was generated across Age, Annual Income, and Spending Score — color-coded by gender — producing a 3×3 matrix of scatter plots and KDE diagonal distributions. This is the most information-dense single visualization in the EDA section.

**Key observation:** The `Annual_Income × Spending_Score` panel reveals approximately **five visually distinct groupings** before any algorithm is applied. This human-detectable structure validates that K-Means will identify real behavioral segments rather than forcing arbitrary partitions onto homogeneous data.

<p align="center">
  <img src="images/pair_plot.png" width="700" alt="Pairwise Feature Relationships — Gender Hue"/>
</p>

---

### 7. Annual Income vs. Spending Score — Primary Business View

This scatter plot is the analytically critical pre-clustering view. When income is plotted against spending score, **five natural groupings emerge visibly** — low income / high spending, high income / high spending, mid income / moderate spending, low income / low spending, and high income / low spending. This visual confirmation validates that the feature space contains genuine latent cluster structure suitable for K-Means.

<p align="center">
  <img src="images/income_vs_spending.png" width="700" alt="Annual Income vs. Spending Score"/>
</p>

---

### 8. Age vs. Spending Score — Demographic Behavior View

Customers aged **20–35 exhibit the widest variance in spending score** — ranging from extremely conservative to extremely impulsive. Customers aged **50+** cluster predominantly at lower spending scores, indicating age-related behavioral convergence toward financial caution. This pattern directly informs age-segmented campaign design.

<p align="center">
  <img src="images/age_vs_spending.png" width="700" alt="Age vs. Spending Score"/>
</p>

---

## 🛠️ Data Preprocessing & Feature Engineering

### Feature Selection

Two parallel feature configurations were prepared to support different analytical objectives:

* **2D Feature Matrix:** `Annual_Income` + `Spending_Score` — the primary clustering configuration for clean, business-interpretable segment visualization.
* **3D Feature Matrix:** `Age` + `Annual_Income` + `Spending_Score` — the extended configuration used for multi-dimensional behavioral profiling and dimensionality reduction inputs.

`CustomerID` was excluded as a non-behavioral administrative key. `Gender` was retained separately for post-hoc demographic analysis within discovered clusters rather than used as a direct clustering input.

### Standard Scaling (Z-Score Normalization)

K-Means calculates **Euclidean distance** between every data point and every cluster centroid. Features with larger numeric ranges — such as `Annual_Income` (spanning ~$122k) — will dominate distance calculations and render lower-range features like `Spending_Score` (spanning 98 points) functionally invisible to the algorithm unless scaling is applied.

`StandardScaler` from `sklearn.preprocessing` was applied to transform all features to zero mean and unit variance:

$$\mathbf{z} = \frac{\mathbf{x} - \mu}{\sigma}$$

**Post-scaling verification:**
* Mean of all scaled features → ~0.0000 ✅
* Standard deviation of all scaled features → ~1.0000 ✅

This normalization guarantees equal feature contribution to all distance calculations, producing unbiased cluster partitions.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled     = scaler.fit_transform(df[['Age', 'Annual_Income', 'Spending_Score']])
X_2d_scaled  = scaler.fit_transform(df[['Annual_Income', 'Spending_Score']])
```

---

## 🤖 K-Means Clustering Theory & Optimization

The core segmentation engine implements the **K-Means Clustering algorithm**, an iterative centroid-based method for partitioning multi-dimensional feature spaces into K non-overlapping groups.

### Mathematical Objective Function

The algorithm minimizes the Within-Cluster Sum of Squares (WCSS) — the total squared Euclidean distance from every data point to its assigned cluster centroid:

$$J = \sum_{j=1}^{K} \sum_{i \in C_j} \| x_i - \mu_j \|^2$$

Where $K$ is the total number of clusters, $C_j$ is the set of points in cluster $j$, $x_i$ is an individual feature vector, and $\mu_j$ is the centroid of cluster $j$. Minimizing $J$ drives the algorithm toward compact, well-separated cluster configurations.

### K-Means Algorithmic Sequence

```
  Step 1 [Initialize]  →  Place K centroids using K-Means++ initialization
  Step 2 [Assign]      →  Each data point joins its nearest centroid (Euclidean distance)
  Step 3 [Update]      →  Recompute each centroid as the mean of its assigned points
  Step 4 [Converge]    →  Repeat Steps 2–3 until centroid positions stabilize
```

### K-Means++ Initialization

Standard random centroid initialization risks placing multiple starting centroids in the same region of the feature space, leading to poor convergence and suboptimal cluster assignments. **K-Means++** resolves this by selecting each subsequent centroid with probability proportional to its squared distance from the nearest already-selected centroid — ensuring initial centroids are spatially well-distributed.

### Final Model Configuration

```python
KMeans(
    n_clusters   = 5,           # Validated by Elbow Method + Silhouette Score
    init         = 'k-means++', # Intelligent spatial initialization
    n_init       = 10,          # 10 independent runs; best result retained
    random_state = 42           # Fixed seed for full reproducibility
)
```

Setting `n_init=10` means the full algorithm is executed 10 independent times with different random seeds. The solution with the lowest final WCSS is retained — guarding against the rare case where even K-Means++ produces a poor initialization.

### Parallel Model Configurations

| Model | Feature Inputs | Primary Purpose |
| :--- | :--- | :--- |
| **2D K-Means** | `Annual_Income` + `Spending_Score` | Business-interpretable 2D cluster scatter with visible centroids |
| **3D K-Means** | `Age` + `Annual_Income` + `Spending_Score` | Multi-dimensional profiling; input for PCA and t-SNE projection |

---

## 🔍 Determining the Optimal Cluster Count ($K$)

Selecting $K$ is the most consequential modeling decision in a K-Means pipeline. An insufficient $K$ over-aggregates meaningfully different customer groups; an excessive $K$ produces segments too small for actionable marketing strategies. Two complementary validation techniques were applied.

### Method 1 — The Elbow Method (WCSS Evaluation)

WCSS was computed for every integer value of $K$ from 1 to 11 by fitting a separate K-Means model at each step. As $K$ increases, WCSS always decreases — more clusters means each individual cluster is tighter. The optimal $K$ is identified at the **inflection point** where marginal WCSS reduction transitions from steep to gradual.

**Result:** The WCSS curve shows a distinct, unambiguous elbow at **$K = 5$**. Beyond this point, adding additional clusters yields only marginal variance reduction with no meaningful increase in analytical value.

<p align="center">
  <img src="images/elbow_method.png" width="700" alt="Elbow Method — WCSS vs. K"/>
</p>

---

### Method 2 — Silhouette Coefficient Validation

The Silhouette Score provides a complementary quality metric by measuring both **cluster compactness** and **cluster separation** simultaneously. For each data point $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\ b(i))}$$

Where $a(i)$ is the mean intra-cluster distance (compactness) and $b(i)$ is the mean nearest-cluster distance (separation). Scores range from $-1$ to $+1$, where values approaching $+1$ indicate tight, well-separated clusters.

Silhouette scores were computed for $K = 2$ through $K = 11$:

**Result:** **$K = 5$ returned the highest Silhouette Score** in the evaluated range, independently confirming the Elbow Method finding. The elevated score at $K = 5$ relative to neighboring values confirms this is a genuine structural optimum rather than a visual artifact of the elbow curve.

<p align="center">
  <img src="images/silhouette_scores.png" width="700" alt="Silhouette Score vs. K"/>
</p>

> ✅ **Dual-method convergence at $K = 5$:** The Elbow Method and Silhouette Coefficient independently identify the same optimal cluster count. This eliminates subjectivity from the $K$ selection and provides high statistical confidence in the 5-segment model.

---

## 📉 Dimensionality Reduction & Manifold Visualization

The 3D feature matrix (Age + Annual Income + Spending Score) cannot be directly plotted. Dimensionality reduction techniques compress the feature space into 2D projections while preserving the most structurally important information, enabling visual validation of cluster separability.

### 1. Principal Component Analysis (PCA)

PCA is a **linear** orthogonal transformation that projects the data onto a new coordinate system where the axes (principal components) are ordered by the fraction of total data variance they explain. It is deterministic — identical inputs always produce identical outputs.

The first two principal components capture a substantial portion of total variance in the 3D feature space, making the 2D projection a geometrically faithful representation. The PCA cluster visualization confirms that the 5 K-Means segments occupy distinct, largely non-overlapping regions of the projected space.

<p align="center">
  <img src="images/pca_clusters.png" width="700" alt="PCA — Cluster Visualization"/>
</p>

<p align="center">
  <img src="images/pca_variance.png" width="500" alt="PCA — Explained Variance per Component"/>
</p>

---

### 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a **non-linear** probabilistic technique that maps high-dimensional proximity relationships into a low-dimensional embedding by matching conditional probability distributions. It is particularly effective at preserving local neighborhood structure — revealing tight intra-cluster compactness that linear methods may compress or obscure.

**Configuration:**

```python
TSNE(
    n_components  = 2,
    perplexity    = 30,     # Effective neighborhood size — optimal for ~200 samples
    max_iter      = 1000,
    random_state  = 42,
    learning_rate = 'auto',
    init          = 'pca'   # PCA-based initialization for stability
)
```

**Important caveat:** Unlike PCA, the axes in a t-SNE embedding carry no direct interpretable meaning. Inter-cluster distances in t-SNE space are not proportional to true feature-space distances. Its role is purely confirmatory — validating that clusters have genuine local structure, not that they exist at specific geometric distances from one another.

<p align="center">
  <img src="images/tsne_clusters.png" width="700" alt="t-SNE — Cluster Visualization"/>
</p>

| Technique | Type | Axes Interpretable | Deterministic | Optimal Use Case |
| :--- | :--- | :---: | :---: | :--- |
| **PCA** | Linear | ✅ Yes | ✅ Yes | Global structure, variance attribution, feature-linked projection |
| **t-SNE** | Non-linear | ❌ No | ❌ No | Local cluster compactness, visual separation confirmation |

Both techniques provide complementary geometric evidence that K-Means identified real behavioral structure rather than imposing arbitrary partitions on homogeneous data.

---

## 👥 Customer Segment Analysis & Behavioral Archetypes

The trained K-Means model identified five customer archetypes based on the Income × Spending Score feature space. Each cluster was profiled using mean statistics across all original features, and a descriptive business label was assigned based on centroid position.

<p align="center">
  <img src="images/kmeans_clusters_2d.png" width="750" alt="K-Means Clustering — 2D Scatter with Centroids"/>
</p>

<p align="center">
  <img src="images/cluster_heatmap.png" width="650" alt="Cluster Profile Heatmap — Average Feature Values"/>
</p>

<p align="center">
  <img src="images/cluster_profiles.png" width="800" alt="Cluster Profiles — Normalized Feature Comparison"/>
</p>

---

### 🎯 Cluster 0 — Sensible Savers *(Mid Income, Low Spending)*

* **Behavioral Dynamics:** Middle-income earners with conservative spending patterns. They visit the mall with regularity but consistently transact below their financial capacity. Purchase decisions are deliberate, research-driven, and oriented toward necessity rather than impulse.
* **Economic Footprint:** Average annual income ~$55k; average spending score ~40/100. The income-to-score gap signals that value-based friction is suppressing spend rather than genuine budget constraints.
* **Strategic Classification:** A persuadable mid-tier segment. The income is present — the right framing and incentive structure can convert latent purchasing capacity into realized transactions.

---

### 💎 Cluster 1 — Premium Customers *(High Income, High Spending)*

* **Behavioral Dynamics:** The mall's highest-value cohort. These are younger high-earners who translate significant income into consistently high mall engagement. Purchase decisions are confident and brand-aware; this group actively seeks premium experiences.
* **Economic Footprint:** Average annual income ~$85k+; average spending score ~82/100. The highest revenue proxy contribution of any segment in the revenue simulation.
* **Strategic Classification:** Core structural revenue assets. Retention investment in this segment yields the highest per-customer ROI of any group. A single churned Premium Customer represents more lost revenue than several Budget Shoppers combined.

---

### 🛒 Cluster 2 — Budget Shoppers *(Low Income, Low Spending)*

* **Behavioral Dynamics:** Cost-conscious shoppers with genuine financial constraints. Low spending scores reflect budget limitations rather than disengagement. This group visits for essentials, responds strongly to price reductions, and prioritizes functional value over experiential or aspirational purchasing.
* **Economic Footprint:** Average annual income ~$25k; average spending score ~20/100. Basket sizes are small; frequency is the lever available for incremental revenue growth.
* **Strategic Classification:** A volume-dependent, low-margin segment best served through low-cost automated marketing focused on visit frequency and discount-driven conversion rather than ticket size uplift.

---

### ⭐ Cluster 3 — Careless Spenders *(Low Income, High Spending)*

* **Behavioral Dynamics:** The most behaviorally anomalous segment. These customers spend at a rate dramatically disproportionate to their income — their spending score rivals the premium segment despite earning at budget-shopper levels. Purchasing is emotionally and socially driven: trend-responsive, impulsive, and FOMO-sensitive.
* **Economic Footprint:** Average annual income ~$25k; average spending score ~78/100. The income-to-score gap is inverted relative to Conservative Elites — here income is low but engagement is exceptionally high.
* **Strategic Classification:** A high-engagement, high-sensitivity segment. BNPL integrations, social media activation, and limited-edition scarcity tactics convert this group's existing behavioral predisposition into increased basket values.

---

### 💰 Cluster 4 — Conservative Elites *(High Income, Low Spending)*

* **Behavioral Dynamics:** High-earning customers who are structurally disengaged from mall spending. The income capacity is present — the behavioral engagement is not. This group is not price-sensitive; they are experience and quality sensitive. They are allocating their discretionary spend elsewhere.
* **Economic Footprint:** Average annual income ~$85k+; average spending score ~17/100. The most significant untapped revenue opportunity in the entire customer base — the income-to-score gap is wider here than in any other segment.
* **Strategic Classification:** The highest-upside re-engagement target. Converting Conservative Elites from a spending score of ~17 to ~50 — without requiring any change in income or footfall volume — would represent the single largest structural revenue unlock available to the business.

---

## 🚀 Tailored Marketing & Business Growth Strategies

Segment-specific marketing strategies were developed directly from each archetype's behavioral and demographic profile. A strategy appropriate for one segment will actively misalign with the psychology of another — cross-segment contamination reduces marketing efficiency.

* **Sensible Savers:**
  * Deploy value-bundle promotions and combo deals that frame spending as financially rational — this segment responds to ROI arguments, not luxury signals.
  * Implement a points-based loyalty program with gradual reward accumulation, giving them a data-driven reason to increase visit frequency.
  * Distribute early-access clearance alerts and cashback offers on purchases above a defined threshold to systematically raise average basket size.

* **Premium Customers:**
  * Launch a formal VIP membership tier with exclusive, tangible privileges — dedicated staff, priority billing, private lounge access, and pre-launch product previews.
  * Execute highly personalized milestone marketing: birthday rewards, anniversary recognitions, and curated newsletters from premium brand partners operating within the mall.
  * Treat retention as a financial priority: the cost of losing one Premium Customer exceeds the acquisition cost of multiple lower-tier customers.

* **Budget Shoppers:**
  * Issue targeted coupon packs and time-limited flash sale access to provide a direct, low-friction incentive for additional visits.
  * Promote dedicated "Best Value" sections with signage that removes the psychological discomfort associated with budget shopping.
  * Introduce student and senior discount programs to address the specific demographic composition of this cluster.

* **Careless Spenders:**
  * Integrate Buy Now Pay Later (BNPL) payment options — eliminating upfront payment barriers for a group already predisposed to spending above their immediate financial capacity.
  * Execute social media influencer campaigns and mall-exclusive limited-edition product drops that activate FOMO-driven purchase decisions.
  * Optimize product placement strategy: position visually compelling, moderately priced items at high-traffic zones including mall entrances, escalator landings, and checkout areas.

* **Conservative Elites:**
  * Initiate a premium re-engagement campaign that leads with recognition and exclusivity rather than discounts — this segment interprets discounting as a signal of low quality, not good value.
  * Market product craftsmanship, heritage, and exclusivity narratives. This group buys into values and story, not price points.
  * Offer complimentary high-value service consultations (personal styling, home décor, technology advisory) to convert passive foot traffic into active, high-value transactions.
  * Deploy a referral reward program — Conservative Elites have affluent social networks, and a successful referral from this segment reliably introduces another high-income prospect.

---

## 💼 Business Insights

The notebook computed a **Revenue Proxy Score** — defined as `(Annual_Income × Spending_Score) / 100` — for each customer record to simulate relative revenue contribution across segments. This metric was aggregated by cluster to rank segments by structural revenue importance.

<p align="center">
  <img src="images/revenue_by_cluster.png" width="700" alt="Estimated Revenue Contribution per Cluster"/>
</p>

<p align="center">
  <img src="images/gender_per_cluster.png" width="700" alt="Gender Distribution per Customer Segment"/>
</p>

### Priority Action Matrix

| Cluster | Label | Revenue Priority | Primary Action |
| :---: | :--- | :---: | :--- |
| **1** | Premium Customers | 🔴 Critical | Protect and retain at all cost — highest revenue per customer |
| **4** | Conservative Elites | 🔴 Critical | Re-engage — highest revenue *upside* due to income-to-score gap |
| **3** | Careless Spenders | 🟡 High | Scale BNPL integration and social media activation |
| **0** | Sensible Savers | 🟡 High | Convert with value framing — income capacity exists |
| **2** | Budget Shoppers | 🟢 Standard | Automate with low-cost discount and frequency campaigns |

---

## 🎨 Visualization Gallery

All visualizations were saved at 150 DPI. The full image set generated by the notebook pipeline is catalogued below:

<p align="center">
  <img src="images/cluster_visualization.png" width="750" alt="Cluster Visualization Overview"/>
</p>
<p align="center"><em>Figure: K-Means segment boundaries visualized across the primary Income × Spending Score feature plane.</em></p>

| Plot | Filename | Description |
| :--- | :--- | :--- |
| Gender Distribution | `gender_distribution.png` | Bar chart + pie chart of gender split across full customer base |
| Age Distribution | `age_distribution.png` | Histogram with mean line + gender-segmented box plot |
| Income Distribution | `income_distribution.png` | Histogram with mean line + gender-segmented box plot |
| Spending Distribution | `spending_distribution.png` | Histogram with mean line + gender-segmented box plot |
| Correlation Heatmap | `correlation_heatmap.png` | Annotated Pearson correlation matrix — all numerical features |
| Pair Plot | `pair_plot.png` | Full pairwise scatter + KDE matrix colored by gender |
| Income vs. Spending | `income_vs_spending.png` | Pre-clustering business view — 5 visual groupings visible |
| Age vs. Spending | `age_vs_spending.png` | Demographic behavior view colored by gender |
| Elbow Method | `elbow_method.png` | WCSS vs. K (range 1–11) with elbow annotation at K=5 |
| Silhouette Scores | `silhouette_scores.png` | Silhouette coefficient vs. K (range 2–11) |
| K-Means 2D Clusters | `kmeans_clusters_2d.png` | Final cluster scatter with inverse-transformed centroids marked |
| Cluster Sizes | `cluster_sizes.png` | Annotated bar chart of customer count per segment |
| PCA Clusters | `pca_clusters.png` | 2D PCA projection with cluster coloring and variance annotation |
| PCA Variance | `pca_variance.png` | Explained variance bar chart per principal component |
| t-SNE Clusters | `tsne_clusters.png` | Non-linear 2D t-SNE embedding with cluster coloring |
| Cluster Heatmap | `cluster_heatmap.png` | Average Age, Income, and Spending Score values per cluster |
| Cluster Profiles | `cluster_profiles.png` | Normalized horizontal bar comparison across all 5 clusters |
| Revenue by Cluster | `revenue_by_cluster.png` | Simulated revenue proxy contribution ranked by segment |
| Gender per Cluster | `gender_per_cluster.png` | Grouped bar chart of gender composition within each cluster |

---

## 🔑 Key Findings

1. **Five behaviorally distinct customer archetypes exist as genuine latent structure** — visually detectable in the Income × Spending Score scatter plot before any algorithm is applied. K-Means formalizes a pattern that human observation already confirms.

2. **$K = 5$ is objectively validated by two independent statistical methods** — the Elbow Method and Silhouette Coefficient independently converge on the same value, eliminating subjectivity from the modeling decision.

3. **High income does not predict high spending** — Cluster 4 (Conservative Elites) directly contradicts the assumption that affluent customers are automatically high spenders. Income and engagement are structurally independent in this dataset.

4. **The Spending Score distribution is near-uniform across 1–99** — the mall has not successfully polarized customer engagement. This represents a marketing effectiveness gap with direct revenue consequences.

5. **Age negatively correlates with Spending Score ($r = -0.33$)** — younger customers exhibit the widest behavioral variance. The 20–35 age band is the most responsive to marketing interventions and carries the highest upside for spend score uplift.

6. **Cluster 3 (Careless Spenders) demonstrates income-independent high engagement** — with income comparable to Budget Shoppers but spending scores rivaling Premium Customers. This behavioral anomaly is driven by emotional and social purchasing motivations rather than financial capacity.

7. **PCA and t-SNE both confirm geometric cluster separability** — the segments identified by K-Means are not statistical artifacts. They have real, physically separate structure in the original multi-dimensional feature space.

---

## 🏁 Final Conclusion

This project demonstrates a complete, reproducible unsupervised machine learning pipeline — from raw demographic records to structured business intelligence — without the use of any labeled training data.

The five-segment customer model produced by this pipeline is not a theoretical exercise. Each segment maps directly to a distinct marketing response, a specific budget allocation priority, and a measurable revenue opportunity. The Conservative Elites segment alone — high-income, structurally disengaged — represents the most immediate and largest revenue opportunity available to the business without increasing foot traffic or acquiring new customers.

The broader methodological contribution of this project is its demonstration that **relatively small, clean datasets contain sufficient behavioral signal to produce commercially actionable intelligence** when the correct analytical framework is applied. Customer segmentation is not a luxury reserved for organizations with millions of transaction records — it is achievable with 200 data points and the right combination of statistical methods.

---

## 🔮 Future Architecture Enhancements

* **Advanced Clustering Algorithms:** Implement DBSCAN (Density-Based Spatial Clustering) and Agglomerative Hierarchical Clustering to benchmark segment quality against K-Means and capture potentially non-spherical cluster geometries.
* **Gaussian Mixture Models (GMM):** Apply soft-assignment probabilistic clustering to handle customers who sit near segment boundaries — assigning partial membership probabilities rather than hard binary assignments.
* **RFM Feature Engineering Integration:** Augment the current demographic features with transaction-derived RFM (Recency, Frequency, Monetary) metrics if purchase history data becomes available, enabling behavioral profiling at a transaction level rather than a demographic level.
* **Predictive Customer Lifetime Value (CLV) Modeling:** Overlay a supervised regression layer on top of the discovered segments to forecast individual CLV from initial demographic inputs, enabling acquisition-stage targeting decisions.
* **Streaming Segmentation Infrastructure:** Integrate real-time data pipelines to dynamically reassign customers to updated cluster models as new behavioral data is captured — replacing the current static snapshot approach.
* **Interactive Stakeholder Dashboard:** Deploy a Streamlit or Dash web application exposing cluster filtering, customer-level lookup, and segment performance metrics to non-technical business stakeholders.

---

<p align="center">
  <strong>DevelopersHub Corporation — Data Science Internship | Task 2</strong><br/>
  <em>Mall Customer Segmentation Using Unsupervised Machine Learning</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square">
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square">
</p>
