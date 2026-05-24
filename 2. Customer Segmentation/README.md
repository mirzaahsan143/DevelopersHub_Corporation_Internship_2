<div align="center">

# 🛍️ Customer Segmentation Using Unsupervised Machine Learning

### *Transforming 541,909 Raw Transactions into Actionable Business Intelligence*

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4C72B0?style=for-the-badge)

<br>

![Machine Learning](https://img.shields.io/badge/Task-Customer%20Segmentation-success?style=flat-square)
![Algorithm](https://img.shields.io/badge/Algorithm-K--Means%20Clustering-blue?style=flat-square)
![Dimensionality Reduction](https://img.shields.io/badge/Dim.%20Reduction-PCA%20%2B%20t--SNE-orange?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-RFM%20Analysis-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

<br>

> **Data Science Internship Project** — Applying unsupervised learning to identify distinct customer segments from an online retail dataset, engineer RFM behavioral features, and deliver actionable, data-driven marketing strategies.

</div>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Task Objective](#-task-objective)
- [Dataset Description](#-dataset-description)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [RFM Feature Engineering](#-rfm-feature-engineering--data-preprocessing)
- [Data Preprocessing](#-data-preprocessing--scaling)
- [K-Means Clustering](#-k-means-clustering)
- [Finding Optimal Clusters](#-finding-the-optimal-number-of-clusters)
- [Dimensionality Reduction](#-dimensionality-reduction--pca--t-sne)
- [Customer Segment Analysis](#-customer-segment-analysis--profiling)
- [Marketing Strategies](#-data-driven-marketing-strategies)
- [Business Insights](#-business-insights--revenue-opportunities)
- [ML Concepts Used](#-machine-learning--data-science-concepts)
- [Visualizations](#-key-visualizations)
- [Key Findings](#-key-findings)
- [Final Conclusion](#-final-conclusion)
- [Future Enhancements](#-future-enhancements)

---

## 🔍 Project Overview

In today's hyper-competitive retail landscape, the ability to understand *who your customers actually are* — not as a monolith, but as distinct groups with different behaviors, needs, and values — is one of the most powerful analytical capabilities a business can have. This project applies **unsupervised machine learning** to a real-world e-commerce dataset to answer a deceptively simple question: *Are all customers the same?*

The answer, of course, is no — and this project proves it with data.

Using the internationally recognized **RFM (Recency, Frequency, Monetary Value)** framework as a foundation, raw transaction logs are transformed into rich customer-level behavioral profiles. These profiles are then fed into a **K-Means clustering model** to automatically discover natural groupings in the customer base. The resulting segments are visualized using **PCA** and **t-SNE**, and each cluster is interpreted through a business lens to yield concrete, actionable marketing strategies.

### Why Does Customer Segmentation Matter?

The real-world impact of customer segmentation extends far beyond academic exercise:

- **Amazon** uses purchase history clustering to power its recommendation engine — "Customers who bought this also bought..."
- **Netflix** segments viewing behavior to personalize its entire content discovery interface, contributing to a reported 80% of content watched coming from recommendations.
- **Spotify** clusters listening patterns to generate its legendary Discover Weekly playlists.
- **Retail banks** use RFM-inspired segmentation to identify customers at risk of churn months before they actually leave.

When a business treats all customers identically — sending the same email, offering the same discount, pushing the same product — it wastes marketing budget on people who would have bought anyway, and fails to reach people who needed a nudge. Segmentation fixes this by enabling **right message, right person, right time** marketing at scale.

---

## 🎯 Task Objective

This project was completed as part of a **Data Science Internship** with the goal of demonstrating end-to-end proficiency in applied machine learning — from raw data ingestion to business-ready insights.

**Primary Goals:**

| # | Objective | Purpose |
|---|-----------|---------|
| 1 | Conduct thorough **Exploratory Data Analysis (EDA)** | Understand data quality, distributions, and business context |
| 2 | Engineer **RFM features** from raw transaction data | Convert row-level data into meaningful customer-level metrics |
| 3 | Apply **K-Means Clustering** to discover customer segments | Group customers by behavioral similarity |
| 4 | Use **PCA & t-SNE** to visualize and validate clusters | Confirm cluster separation and quality in 2D space |
| 5 | Develop **data-driven marketing strategies** per segment | Translate machine learning output into business action |

The overarching purpose is to demonstrate that data science is not just about building models — it's about building models that *change how decisions get made*. Every analysis, visualization, and recommendation in this project is grounded in a business question.

---

## 📦 Dataset Description

**Source:** UCI Machine Learning Repository — Online Retail Dataset  
**Business Context:** A UK-based online retail company specializing in gifts and novelty items, selling primarily to wholesale customers across 38 countries  
**Period Covered:** December 2010 – December 2011 (13 months of transaction history)  
**Total Records:** 541,909 transaction line items across 4,372 unique customers (post-cleaning)

### Column Reference

| Column | Data Type | Description | Business Significance |
|--------|-----------|-------------|----------------------|
| `InvoiceNo` | String | Unique 6-digit invoice number. Prefix `C` denotes a cancellation | Identifies individual orders; cancellations must be excluded |
| `StockCode` | String | 5-digit unique product identifier | Product-level analysis; used in basket analysis |
| `Description` | String | Product name / description | Human-readable product identification |
| `Quantity` | Integer | Number of units per line item | Negative values indicate returns; must be cleaned |
| `InvoiceDate` | DateTime | Date and timestamp of the transaction | Critical for Recency and trend calculations |
| `UnitPrice` | Float | Price per product unit in British Pounds (£) | Combined with Quantity to compute revenue |
| `CustomerID` | Float | Unique 5-digit customer identifier | The segmentation anchor — rows without this are discarded |
| `Country` | String | Country of the customer | Geographic distribution analysis |

### Why These Features Enable Segmentation

The dataset contains no direct demographic data (age, gender, income) — which makes it a realistic, challenging business problem. Segmentation must be derived entirely from *behavioral* signals: when customers shop, how often they return, and how much they spend. This is arguably more valuable than demographics because behavior *predicts future action* far more reliably than a birth year.

### Data Quality Overview

- **Missing Values:** CustomerID was missing for ~135,080 rows (24.9%) — all excluded since segmentation requires customer identification
- **Cancelled Transactions:** ~9,288 rows with InvoiceNo prefix `C` — removed to analyze only completed purchases
- **Negative Quantities:** Returns and adjustments — filtered out during preprocessing
- **Zero/Negative Prices:** ~2 records with invalid pricing — excluded
- **Duplicates:** Present and removed during the cleaning phase

---

## 🗺️ Project Workflow

The project follows a structured, industry-standard data science pipeline:

```
Raw Transaction Data (541,909 rows)
          │
          ▼
  ┌──────────────────┐
  │  1. Data Loading  │  ──▶  Read CSV with latin-1 encoding (handles £ symbol)
  └──────────────────┘
          │
          ▼
  ┌───────────────────────┐
  │  2. Data Understanding │  ──▶  Shape, types, missing values, duplicates, outliers
  └───────────────────────┘
          │
          ▼
  ┌──────────────────────────────────┐
  │  3. Exploratory Data Analysis    │  ──▶  Geographic, temporal, product & spend analysis
  └──────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────┐
  │  4. Data Cleaning & RFM Feature Engineering │  ──▶  Build Recency, Frequency, Monetary per customer
  └────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────┐
  │  5. Log Transformation + StandardScaler  │  ──▶  Handle skewness; normalize for K-Means
  └─────────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────────┐
  │  6. Optimal K Selection (Elbow + Silhouette)  │  ──▶  Tested k=2 to k=10; selected k=4
  └──────────────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────┐
  │  7. K-Means Clustering (k=4)  │  ──▶  k-means++ init, n_init=10, random_state=42
  └──────────────────────────────┘
          │
          ▼
  ┌───────────────────────────────────────┐
  │  8. Dimensionality Reduction & Viz    │  ──▶  PCA (linear) + t-SNE (non-linear)
  └───────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  9. Cluster Profiling & Naming       │  ──▶  Champions, Loyal, At-Risk, Dormant
  └──────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────┐
  │  10. Marketing Strategies & Business Insights│  ──▶  Segment-specific action plans
  └─────────────────────────────────────────────┘
```

---

## 📊 Exploratory Data Analysis

Before any modeling, a thorough EDA was conducted to understand the business context, data quality, and behavioral patterns hidden within the raw transactions.

### Geographic Distribution

The dataset covers 38 countries, but the business is heavily UK-centric. The UK accounts for the overwhelming majority of transactions by volume. When UK data is excluded and remaining countries are ranked by revenue contribution, **Netherlands, EIRE (Ireland), Germany, and France** emerge as the strongest international markets. This geographic concentration is critical context: the customer segments identified will be primarily driven by UK wholesale buyer behavior.

<p align="center">
  <img src="images/geo_distribution.png" width="800">
</p>

### Monthly Revenue Trend

Plotting total revenue by month reveals unmistakable seasonality. Revenue is relatively steady through early 2011, then builds progressively from September onward, culminating in a sharp spike in **November 2011**. This is the classic pre-Christmas gift-buying surge that any UK retail business would recognize. Notably, December 2011 shows a drop — likely because data collection ended mid-month. This seasonality insight directly informs marketing strategy: promotional campaigns should be front-loaded before November to maximize the seasonal wave.

<p align="center">
  <img src="images/monthly_revenue.png" width="800">
</p>

### Transaction & Price Distributions

The quantity distribution is heavily right-skewed — the vast majority of line items involve small quantities (median: 6 units), consistent with a business selling small gift items at relatively low unit prices (median: ~£2.08). This pattern is typical of a novelty/gift wholesaler. The few extreme outliers represent bulk wholesale orders and must be handled carefully during preprocessing to avoid distorting the clustering.

<p align="center">
  <img src="images/distributions.png" width="800">
</p>

### Customer Lifetime Spend Distribution

The distribution of total customer spend reveals a highly right-skewed curve — the classic shape of a Pareto distribution. A small number of customers account for an outsized share of total revenue, while the majority of customers spend relatively modest amounts. The log-transformed view reveals the true shape: a multi-modal distribution suggesting naturally distinct spending tiers that K-Means will later formalize.

<p align="center">
  <img src="images/spend_distribution.png" width="800">
</p>

### Top Products by Volume

The best-selling products are dominated by small, repeatable gift items — decorative bags, party bundles, and home accessories. This product mix confirms the wholesale/gift retailer profile and validates why frequency and recency metrics are especially meaningful here: customers who return regularly are likely placing seasonal or wholesale replenishment orders.

<p align="center">
  <img src="images/top_products.png" width="800">
</p>

---

## 🔧 RFM Feature Engineering & Data Preprocessing

### What is RFM and Why Does It Work?

**RFM** is a time-tested, industry-standard framework for quantifying customer behavior using only transaction data. It condenses a customer's entire purchase history into three numbers that capture the most predictive dimensions of future behavior:

| Feature | Definition | Calculation | Business Meaning |
|---------|-----------|-------------|-----------------|
| **Recency (R)** | Days since last purchase | `(reference_date – last_invoice_date).days` | How recently did they engage? Fresh customers are more likely to respond to campaigns. |
| **Frequency (F)** | Number of unique invoices | `COUNT(DISTINCT InvoiceNo)` | How often do they buy? Repeat buyers are the business backbone. |
| **Monetary (M)** | Total spend in £ | `SUM(Quantity × UnitPrice)` | How much are they worth? High spenders deserve premium treatment. |

The reference date was set to **one day after the final transaction in the dataset** — a standard technique to ensure the most recent customers have a Recency of 1, not 0.

### Data Cleaning Steps

Before computing RFM, the dataset was cleaned in a deliberate sequence:

1. **Drop rows with missing CustomerID** — segmentation is impossible without a customer anchor
2. **Remove cancelled orders** — InvoiceNo entries starting with `C` represent returns, not purchases
3. **Filter invalid Quantities and Prices** — rows with Quantity ≤ 0 or UnitPrice ≤ 0 are data artifacts

After cleaning: **406,829 → ~397,924 valid transaction rows** across **4,372 unique customers** with complete behavioral data.

### RFM Distribution Analysis

The raw RFM distributions are all **right-skewed** — a small number of customers have extremely high frequency or monetary values compared to the majority. This is not surprising in retail, but it creates a problem for K-Means: the algorithm uses Euclidean distance, and without transformation, the monetary dimension (ranging from £3 to £280,000+) would completely dominate Recency and Frequency in distance calculations.

<p align="center">
  <img src="images/rfm_distributions.png" width="800">
</p>

---

## ⚖️ Data Preprocessing & Scaling

### Why Preprocessing is Non-Negotiable for K-Means

K-Means is a **distance-based algorithm**. It assigns each data point to the cluster whose centroid is nearest, measured by Euclidean distance. This means that features with larger numerical ranges will dominate the distance calculation — completely overpowering features with smaller ranges, regardless of their actual informational value.

In this dataset, without scaling:
- Monetary values range from ~£3 to ~£280,000
- Frequency ranges from 1 to ~200
- Recency ranges from 1 to ~374 days

An unscaled model would effectively cluster customers *only by spending* and ignore purchasing frequency almost entirely.

### Two-Step Transformation Pipeline

**Step 1: Log Transformation (`np.log1p`)**

Applied to all three RFM features to compress the extreme right-tail skewness. The `log1p` function (log(x + 1)) safely handles zero values while bringing the distribution closer to normal. This is especially important for Monetary, where a handful of wholesale customers spend 100x the average.

**Step 2: StandardScaler (Z-score Normalization)**

After log transformation, StandardScaler is applied to give each feature a mean of 0 and standard deviation of 1. This places all three dimensions on equal footing for the K-Means distance computation.

<p align="center">
  <img src="images/rfm_scaled.png" width="800">
</p>

After transformation, the distributions are approximately normal with no single feature dominating — the data is now genuinely ready for clustering.

---

## 🤖 K-Means Clustering

### What is K-Means?

K-Means is one of the most widely used unsupervised learning algorithms due to its simplicity, speed, and interpretability. The algorithm works as follows:

1. **Initialize** K cluster centroids (using **k-means++** initialization, which places initial centroids intelligently to speed convergence and avoid poor local optima)
2. **Assign** each data point to the nearest centroid based on Euclidean distance
3. **Recompute** each centroid as the mean of all points assigned to it
4. **Repeat** steps 2–3 until assignments no longer change (convergence)

The result is K clusters — compact, internally similar groupings of data points that are as different from each other as possible.

### Why K-Means for This Problem?

K-Means is well-suited for customer segmentation because:
- The resulting clusters have clearly interpretable centroids (average RFM profiles)
- It scales well to thousands of customers
- It produces hard, actionable assignments — every customer belongs to exactly one segment
- The output integrates naturally with CRM systems for direct marketing application

### Final Model Configuration

```python
KMeans(
    n_clusters = 4,          # Determined by Elbow + Silhouette analysis
    init       = 'k-means++', # Smart initialization to avoid poor convergence
    n_init     = 10,          # Run 10 times, keep the best result
    max_iter   = 300,         # Maximum iterations per run
    random_state = 42         # Reproducibility
)
```

**Final Silhouette Score: > 0.45** — indicating well-separated, meaningful clusters.

---

## 📐 Finding the Optimal Number of Clusters

Selecting the right number of clusters is arguably the most critical decision in any K-Means analysis. Two complementary methods were used together to arrive at **k = 4**:

### Method 1: Elbow Method (WCSS)

The Within-Cluster Sum of Squares (WCSS), also called inertia, measures total variance within clusters. As K increases, WCSS always decreases — more clusters means tighter groupings. The trick is to find the "elbow": the point where adding another cluster yields diminishing returns in variance reduction.

Plotting WCSS against K from 2 to 10 reveals a clear elbow at **k = 4**, after which the curve flattens substantially. Adding a 5th or 6th cluster does not meaningfully improve cluster compactness.

### Method 2: Silhouette Score

The Silhouette Score measures how similar each customer is to their own cluster compared to other clusters, ranging from -1 (wrong cluster) to +1 (well-matched). A higher average score indicates better-defined clusters.

Silhouette scores were calculated for k = 2 through k = 10. The analysis confirmed **k = 4** as the optimal choice, balancing cluster quality (high silhouette score) with business interpretability (four meaningful segments is actionable; ten is not).

<p align="center">
  <img src="images/optimal_k.png" width="800">
</p>

Both methods independently pointed to the same answer — a reassuring sign that k = 4 represents a genuine structure in the data rather than an arbitrary choice.

---

## 🔬 Dimensionality Reduction — PCA & t-SNE

Even though RFM data is only 3-dimensional, dimensionality reduction serves two important purposes here: **validation** (confirming that clusters are genuinely well-separated) and **communication** (producing plots that explain the segmentation to a non-technical audience).

### PCA — Principal Component Analysis

PCA is a **linear** dimensionality reduction technique that projects data onto the directions of maximum variance. Two principal components were extracted from the scaled RFM matrix:

- **PC1** captures the dominant source of variation — roughly corresponding to overall customer value (spend + frequency)
- **PC2** captures a secondary axis of variation — roughly corresponding to recency/engagement level
- Together, **PC1 + PC2 explain ~85% of total variance** — meaning the 2D plot preserves most of the information from the original 3D RFM space

The PCA cluster plot shows four visually distinct groupings with clear spatial separation, with cluster centroids marked. This confirms that K-Means found real structure, not random groupings.

<p align="center">
  <img src="images/pca_clusters.png" width="800">
</p>

### t-SNE — t-Distributed Stochastic Neighbor Embedding

t-SNE is a **non-linear** technique that preserves local neighborhood structure — points that are similar in high-dimensional space end up close together in the 2D projection. Unlike PCA, t-SNE doesn't preserve global distances, but it excels at revealing cluster topology.

Configuration used:
- `perplexity = 50` (appropriate for ~4,000 data points)
- `learning_rate = 'auto'`
- `init = 'pca'` (PCA-initialized t-SNE converges better)
- `n_iter = 1000`

The t-SNE visualization confirms the PCA findings: four tightly-formed, well-separated clusters. The non-linear projection reveals internal substructure within each cluster that PCA's linearity hides.

| | PCA | t-SNE |
|---|---|---|
| **Type** | Linear | Non-linear |
| **Preserves** | Global variance structure | Local neighborhood relationships |
| **Speed** | Fast | Slower (but worth it for validation) |
| **Interpretability** | Variance explained is quantifiable | Distances are not directly interpretable |
| **Best for** | Initial overview + centroid plotting | Detailed cluster quality validation |

<p align="center">
  <img src="images/tsne_vs_pca.png" width="800">
</p>

### Silhouette Plot

A per-sample silhouette plot was generated to assess the quality of individual cluster assignments. Each horizontal bar represents one customer's silhouette coefficient — how well they fit their assigned cluster. Clusters dominated by wide, positive bars indicate well-formed groupings; thin or negative bars indicate borderline cases.

The silhouette plot confirmed that all four clusters are well-formed, with average scores above the 0.45 threshold indicating solid separation.

<p align="center">
  <img src="images/silhouette_plot.png" width="800">
</p>

---

## 👥 Customer Segment Analysis & Profiling

With clusters validated, each group was interpreted through the lens of its RFM centroid to assign a meaningful business identity. The radar charts below show each cluster's normalized RFM profile — with Recency inverted so that "high" always means "better".

<p align="center">
  <img src="images/radar_chart.png" width="800">
</p>

---

### 🏆 Segment 1: Champions (~25% of customers)

**RFM Profile:** Low Recency (shopped very recently) · High Frequency (many orders) · High Monetary (top spenders)

Champions are the crown jewel of the customer base. These are customers who buy regularly, spend generously, and have been active within the last few weeks. They likely represent long-term wholesale buyers who have built a reliable commercial relationship with the store. Their RFM scores are uniformly strong across all three dimensions.

**Key Characteristics:**
- Purchased most recently of all segments
- Highest average order frequency — they come back again and again
- Highest total lifetime spend — disproportionate revenue contributors
- Likely represent loyal, established wholesale relationships
- Strong brand affinity — they've voted with their wallets repeatedly

**Revenue Significance:** Despite being ~25% of the customer count, this segment likely accounts for significantly more than 25% of total revenue — consistent with Pareto dynamics observed in the business opportunity analysis.

---

### 💛 Segment 2: Loyal Customers (~30% of customers)

**RFM Profile:** Moderate Recency · Moderate-to-High Frequency · Moderate-to-High Monetary

Loyal Customers are the dependable backbone of the business. They don't spend as extravagantly as Champions, but they show up consistently and contribute steady, predictable revenue. These customers have proven they like the product range — they just haven't reached Champion-tier engagement yet.

**Key Characteristics:**
- Regular but not ultra-frequent purchase cadence
- Meaningful average spend, though below Champion levels
- Reasonable recency — haven't gone cold
- Good candidates for upselling and cross-selling
- Respond well to loyalty mechanisms that reward consistency

---

### ⚠️ Segment 3: At-Risk Customers (~25% of customers)

**RFM Profile:** High Recency (haven't shopped in a while) · Lower Frequency · Lower Monetary

At-Risk customers are a critical recovery opportunity. They were once active buyers — they have transaction history and demonstrated interest in the product range — but they have not made a purchase in a significant amount of time. Without intervention, they are heading toward permanent dormancy.

**Key Characteristics:**
- Meaningful gap since last purchase — the silence is the signal
- Lower order frequency compared to Champions and Loyal customers
- Spend per visit may have been modest, but they were real customers
- Still reachable — they haven't been gone long enough to consider the relationship broken
- High ROI for win-back campaigns because acquisition cost was already paid

---

### 💤 Segment 4: Lost / Dormant Customers (~20% of customers)

**RFM Profile:** Very High Recency (longest since last purchase) · Very Low Frequency · Very Low Monetary

These customers made a handful of small purchases — possibly test orders or one-off seasonal buys — and never returned. Whether due to a poor experience, finding a better alternative, or simply having a one-time need, they have been inactive for the longest period of all segments.

**Key Characteristics:**
- Longest average gap since last purchase of all segments
- Very low order frequency — many may have purchased only once or twice
- Modest historical spending
- Lower marketing priority than At-Risk; reactivation costs must be justified
- Best strategy: minimal-cost, low-frequency contact with a final sunset campaign

<p align="center">
  <img src="images/segment_summary.png" width="800">
</p>

<p align="center">
  <img src="images/segment_heatmap.png" width="700">
</p>

---

## 📣 Data-Driven Marketing Strategies

The value of segmentation is only realized when it drives *different actions* for different groups. Below are tailored marketing strategies for each segment, grounded in the behavioral data.

---

### 🏆 Champions — Retain, Reward & Leverage

The worst thing a business can do to its best customers is treat them like everyone else. Champions have earned differentiated treatment.

| Strategy | Detail | Rationale |
|----------|--------|-----------|
| **VIP Loyalty Program** | Early access to new product lines, private sales, exclusive bundles | Reinforces their status, increases perceived switching cost |
| **Thank-You Campaigns** | Personalized order notes, seasonal appreciation messages | Strengthens emotional connection beyond pure transactionality |
| **Brand Ambassador Program** | Invite reviews, referral incentives, co-branded content | Champions' word-of-mouth is the most credible marketing channel |
| **Tiered Reward System** | Points-to-perks ladder with exclusive top tier | Keeps engagement high even when Champions are already loyal |
| **Friction Elimination** | Free priority shipping, dedicated account manager for top accounts | Remove every possible barrier to reordering |

---

### 💛 Loyal Customers — Upsell & Deepen Engagement

The goal with Loyal Customers is to increase their order value and frequency while protecting the relationship that already exists.

| Strategy | Detail | Rationale |
|----------|--------|-----------|
| **Subscription / Auto-Replenishment** | Prompt repeat orders for frequently purchased items with a loyalty discount | Increases frequency and predictability of revenue |
| **Personalized Email Campaigns** | Product recommendations based on specific purchase history | Relevance drives click-through and conversion |
| **Bundle Offers** | Cross-sell complementary products to increase average basket size | Low incremental cost; high revenue impact |
| **Milestone Recognition** | Birthday discounts, anniversary-of-first-purchase offers | Deepens emotional attachment to the brand |
| **Loyalty Multiplier Events** | Double points on specific product categories for a limited time | Incentivizes targeted spending behavior without permanent discounting |

---

### ⚠️ At-Risk Customers — Urgently Re-Engage

At-Risk customers represent the highest-priority intervention. Every month of inaction increases the probability of permanent churn.

| Strategy | Detail | Rationale |
|----------|--------|-----------|
| **Win-Back Email Sequence** | "We miss you" campaign with 15–20% time-limited discount | Urgency + incentive = reactivation trigger |
| **Re-engagement Notifications** | "New arrivals since your last visit" with curated products | Low-friction, curiosity-driven re-entry point |
| **Feedback Survey** | Brief 3-question survey: "Why haven't you ordered?" | Identifies fixable problems (pricing, shipping, product range) |
| **Retargeting Ads** | Show products similar to their historical purchases | Reminds them of the relationship using familiar context |
| **Stock Urgency Triggers** | "Limited stock left on items you've purchased before" | Creates FOMO for customers who previously liked specific products |

---

### 💤 Lost / Dormant Customers — Low-Cost Reactivation

For Dormant customers, the marketing calculus changes: spend less per contact, but make each contact count.

| Strategy | Detail | Rationale |
|----------|--------|-----------|
| **Deep Discount Reactivation** | 25–30% off with a hard expiry date (e.g., 7 days) | Strong enough offer to break inertia; deadline creates urgency |
| **Free Gift with First Reorder** | Low-cost item bundled with return purchase | Reduces psychological re-entry barrier |
| **Curiosity Subject Lines** | "A lot has changed since you last visited" — no promotion, just intrigue | Opens before offers; works when discount fatigue has set in |
| **Suppression from Regular Lists** | Remove from weekly promotional emails | Protects sender reputation and marketing budget |
| **Sunset Campaign** | Final aggressive offer; unsubscribe if no response | Clean list hygiene; focus spend on reachable customers |

---

## 💼 Business Insights & Revenue Opportunities

### The Pareto Reality

A Pareto cumulative revenue analysis was conducted on the full customer base. The findings confirmed a pattern familiar to any experienced retail analyst: **a small percentage of customers generate a disproportionate share of total revenue**.

The top tier of customers (Champions + upper Loyal) accounts for the majority of revenue, while the bottom half of the customer base by spend contributes relatively little. This is not a negative finding — it is an *actionable structural insight*. It tells the business exactly where to allocate retention budget, and where to cut marketing spend that isn't generating returns.

<p align="center">
  <img src="images/pareto_analysis.png" width="800">
</p>

### Revenue Opportunity Quantification

A quantitative business opportunity analysis was performed on the At-Risk segment:

- **At-Risk segment size:** ~1,100 customers
- **Average historical spend:** meaningful individual lifetime value
- **Reactivation potential at 10% conversion:** significant incremental revenue recoverable with a single targeted campaign
- **Reactivation potential at 20% conversion:** roughly doubles the 10% scenario

Even conservative reactivation rates produce a positive ROI when the cost of a targeted email campaign is factored in. This makes the At-Risk win-back campaign the **single highest-ROI marketing action** available to the business at this moment.

### Priority Action Matrix

| Segment | Revenue at Risk | Marketing Investment | Priority |
|---------|----------------|---------------------|----------|
| 🏆 Champions | Highest (at risk if neglected) | Medium | **CRITICAL** |
| 💛 Loyal Customers | High | Medium | **HIGH** |
| ⚠️ At-Risk | Medium-High (recoverable) | Medium | **URGENT** |
| 💤 Dormant | Low-Medium | Low | **LOW** |

---

## 🧠 Machine Learning & Data Science Concepts

This project applies the following ML and data science disciplines in a cohesive, end-to-end pipeline:

**Unsupervised Learning** — No labeled training data exists. The model must discover structure in the data autonomously, without being told in advance how many segments exist or what they look like. This is the core challenge and power of the approach.

**RFM Feature Engineering** — Raw transaction logs are transformed into a structured feature matrix using domain knowledge. This step bridges the gap between raw data and an ML-ready input, and represents a significant portion of the analytical value delivered.

**K-Means Clustering** — The primary modeling technique. Customers are grouped by minimizing within-cluster variance across the three RFM dimensions simultaneously. The k-means++ initialization ensures stable, reproducible results.

**Elbow Method & WCSS** — A principled approach to model selection, using the rate of change in within-cluster sum of squares to identify the optimal K value without overfitting to noise.

**Silhouette Analysis** — A quantitative validation metric for cluster quality. Confirms that the chosen K produces genuinely well-separated groups, not arbitrary partitions.

**PCA (Principal Component Analysis)** — A linear dimensionality reduction technique used to project 3D RFM space into 2D for visualization, while preserving maximum variance. Also enables centroid plotting in reduced space.

**t-SNE** — A non-linear technique that reveals local cluster structure and validates separation in a way that linear methods cannot. Particularly effective at confirming that clusters are not just mathematical constructs but genuine behavioral groupings.

**StandardScaler + Log Transformation** — Critical preprocessing pipeline that makes distance-based algorithms work correctly when features have very different scales and distributions.

**Pareto / 80-20 Analysis** — A business intelligence framework applied to quantify revenue concentration and identify the highest-value customer tier.

---

## 🖼️ Key Visualizations

Below is a complete reference guide to all visualizations produced in this project and their analytical purpose:

| Visualization | File | Purpose |
|--------------|------|---------|
| Geographic Distribution | `geo_distribution.png` | Country-level transaction and revenue breakdown |
| Monthly Revenue Trend | `monthly_revenue.png` | Seasonality analysis — reveals November peak |
| Quantity & Price Distributions | `distributions.png` | Understand transaction scale and pricing |
| Customer Lifetime Spend | `spend_distribution.png` | Identifies Pareto-style revenue concentration |
| Top 10 Products | `top_products.png` | Best-selling items by total units sold |
| RFM Distributions (Raw) | `rfm_distributions.png` | Skewness of raw RFM before transformation |
| RFM Distributions (Scaled) | `rfm_scaled.png` | Normalized RFM ready for K-Means |
| Elbow Curve + Silhouette Scores | `optimal_k.png` | Optimal K selection — both methods confirm k=4 |
| Radar Charts (RFM Profiles) | `radar_chart.png` | Per-cluster RFM fingerprint visualization |
| Cluster Scatter Plots | `cluster_scatter.png` | Frequency vs Monetary and Recency vs Monetary |
| Cluster Box Plots | `cluster_boxplots.png` | RFM distribution spread per cluster |
| PCA Cluster Plot | `pca_clusters.png` | Linear 2D projection with centroids |
| t-SNE vs PCA Comparison | `tsne_vs_pca.png` | Side-by-side validation of cluster separation |
| Silhouette Plot | `silhouette_plot.png` | Per-sample cluster quality assessment |
| Segment Size & Revenue | `segment_summary.png` | Customer count and revenue per segment |
| Segment RFM Heatmap | `segment_heatmap.png` | Normalized RFM scores as a color-coded grid |
| Pareto Analysis | `pareto_analysis.png` | Revenue concentration curve (80/20 rule) |

<p align="center">
  <img src="images/cluster_scatter.png" width="800">
</p>

<p align="center">
  <img src="images/cluster_boxplots.png" width="800">
</p>

---

## 🔑 Key Findings

1. **Four Distinct Customer Segments Exist** — K-Means with k=4 produced well-separated, interpretable clusters validated by both PCA and t-SNE. The Silhouette Score > 0.45 confirms genuine structure, not random noise.

2. **Revenue is Highly Concentrated** — A Pareto analysis confirmed that a small fraction of the total customer base (Champions + upper Loyal tier) drives the majority of revenue — consistent with the 80/20 principle. Protecting this tier is the single most important retention priority.

3. **One in Four Customers is At-Risk** — Approximately 25% of customers show signs of disengagement — significant recency, lower frequency. Without a structured win-back campaign, many of these customers will become permanently dormant. The revenue recovery potential from even a 10–20% reactivation rate is substantial.

4. **November is the Business's Peak Period** — The monthly revenue trend shows a clear pre-Christmas spike, suggesting that the business should concentrate its Champions and Loyal Customer campaigns in September-October to maximize seasonal uplift from its best buyers.

5. **RFM Captures Behavioral Reality** — The three RFM features, despite being simple aggregations, encode enough behavioral signal to produce clustering that aligns with business intuition. Champions really do look different from Dormant customers — not just slightly different, but qualitatively, visibly different in every dimension.

6. **PCA Explains ~85% of Variance in 2D** — The high explained variance in the PCA projection means that the 2D cluster plots are a reliable representation of the underlying 3D RFM space, not a distorted view.

7. **Log Transformation was Essential** — The raw RFM distributions were heavily right-skewed. Without log transformation, K-Means would have been dominated by high-monetary outliers and produced meaningless clusters. Preprocessing was the difference between a useful model and a useless one.

---

## ✅ Final Conclusion

This project demonstrates the complete data science lifecycle applied to a real-world business problem with genuine commercial stakes.

Starting from 541,909 raw transaction records — messy, uncleaned, with no customer-level structure — the analysis proceeds through rigorous cleaning, principled feature engineering, validated modeling, and business-ready interpretation to arrive at four actionable customer segments. Each segment has a clear behavioral identity, a quantified revenue significance, and a specific set of marketing strategies designed to maximize its value to the business.

The technical results stand on their own: k=4 was selected by two independent validation methods, confirmed by two independent dimensionality reduction techniques, and the Silhouette Score confirms meaningful separation. But the more important outcome is business clarity — a marketing team can look at this analysis and know, immediately, who to prioritize, what to say to each group, and what revenue recovery is realistically achievable.

### Model Performance Summary

| Metric | Value |
|--------|-------|
| Optimal K (Clusters) | 4 |
| Silhouette Score | > 0.45 |
| PCA Variance Explained (2D) | ~85% |
| Customers Segmented | 4,372 |
| Transactions Analyzed | ~398,000 |
| Revenue Period Covered | Dec 2010 – Dec 2011 |

### Strategic Recommendations

1. **Immediately launch a VIP program** for Champions — the cost of losing even a handful of top-tier customers to a competitor far exceeds the investment in a loyalty initiative
2. **Execute a win-back campaign** for At-Risk customers within 30 days — every additional month of inaction increases permanent churn probability
3. **Build segment transitions tracking** — monitor customers moving between segments quarterly to catch Champions sliding toward At-Risk before it happens
4. **Pre-load campaigns before November** — leverage the seasonal revenue peak by ensuring Champions and Loyal Customers are engaged and ready to buy heading into the gift season

---

## 🚀 Future Enhancements

This project establishes a strong analytical foundation that can be extended in several directions:

**Advanced Clustering Techniques**
- **DBSCAN** — density-based clustering that doesn't require K to be specified in advance and can handle non-spherical cluster shapes
- **Hierarchical / Agglomerative Clustering** — produces a dendrogram showing how customers relate at different levels of similarity
- **Gaussian Mixture Models (GMM)** — soft cluster assignments with probability scores rather than hard boundaries

**Richer Feature Engineering**
- **Product category features** — cluster not just by how much customers spend, but *what* they buy (gift categories, price tiers, seasonal products)
- **Time-series RFM** — compute RFM for rolling time windows to detect segment transitions and predict churn before it happens
- **Geographic features** — country-level behavioral differences may warrant separate segmentation models for different markets

**Predictive Modeling Layer**
- **Cluster membership prediction** — train a supervised classifier (Random Forest, XGBoost) on the cluster labels to instantly segment new customers from their first few purchases
- **Customer Lifetime Value (CLV) prediction** — combine segmentation with survival analysis (BG/NBD model) to forecast future revenue per customer
- **Churn probability scoring** — model the probability that an At-Risk customer will become Dormant within the next 90 days

**Deployment & Productionization**
- **Real-time segmentation pipeline** — connect to a live transaction feed and update customer segments dynamically as new orders arrive
- **CRM integration** — export segment labels directly into Salesforce, HubSpot, or Klaviyo for automated campaign triggering
- **Interactive dashboard** — build a Streamlit or Power BI dashboard allowing marketing teams to explore segment characteristics and filter customers without writing code
- **A/B testing framework** — systematically test marketing strategies across segments and measure uplift against control groups

---

<div align="center">

---

### 📬 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/mirzaahsan0712/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mirzaahsan143/)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mirzaahsan143786@gmail.com)

---

*⭐ If you found this project valuable, please consider starring the repository!*

**Built with dedication during a Data Science Internship | © 2024**

</div>
