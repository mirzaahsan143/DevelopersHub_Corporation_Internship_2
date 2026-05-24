# 🛍️ E-Commerce Customer Segmentation Engine
### Advanced Behavioral Analytics & Unsupervised Machine Learning Pipeline
<p align="center">
  <img src="https://img.shields.io/badge/Data%20Science-Enterprise%20Analytics-blue?style=for-the-badge&logo=python&logoColor=white" alt="Data Science">
  <img src="https://img.shields.io/badge/ML-Unsupervised%20Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Clustering-K--Means%20%7C%20RFM-green?style=for-the-badge" alt="K-Means">
  <img src="https://img.shields.io/badge/Manifold%20Learning-PCA%20%7C%20t--SNE-red?style=for-the-badge" alt="PCA & t-SNE">
</p>

---

## 📌 Project Overview
In modern data-driven retail systems, understanding target demographics and purchasing habits is crucial for maximizing marketing performance and driving revenue development. This project implements an **unsupervised machine learning framework** centered on **K-Means Clustering** to segment customers of an online retail business based on explicit transaction histories.

Unsupervised learning is essential for discovering hidden historical structures within multi-dimensional datasets without requiring manual classification labels. By engineering structural features derived from raw transactional documents, this algorithm reveals core patterns of user engagement.

### Real-World Business Value
* **Hyper-Targeted Marketing Campaigns:** Companies maximize return on investment (ROI) by optimizing campaign delivery to specific customer groups rather than deploying a blanket approach.
* **Optimized Retention Frameworks:** Predicting and preventing churn through personalized reward structures and customized communications directly decreases customer acquisition costs.
* **Product Recommendations & Up-Selling:** Modern retail leaders use granular data segments to tailor homepages, bundle products, and cross-sell effectively.

---

## 🎯 Task & Internship Objectives
The primary business and technical goal of this analysis is to transform raw, noisy e-commerce invoice records into operational business value.

* **Deep Exploratory Data Analysis (EDA):** Identify and adjust for systemic patterns, dataset anomalies, structural skews, and historical trends within raw transaction inputs.
* **Feature Engineering:** Extract meaningful metrics using the industry-standard RFM (Recency, Frequency, Monetary) matrix to model economic relationships per user identity.
* **K-Means Clustering:** Configure an optimized K-Means algorithm to partition the consumer base into homogeneous clusters with distinct purchasing behaviors.
* **Dimensionality Reduction:** Utilize Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) to transform hyper-dimensional configurations into scannable vector landscapes.
* **Strategic Recommendations:** Convert statistical descriptions into clear business guidance and targeted marketing programs.

---

## 📊 Dataset Description & Data Schema
The analysis utilizes the **Online Retail Dataset** sourced from the UCI Machine Learning Repository, capturing real-world international transactions for a UK-based online retail storefront over a 12-month period.

* **Temporal Coverage:** December 1, 2010 – December 9, 2011.
* **Initial Shape:** 541,909 rows × 8 features.

### Features & Structural Typology

| Variable Name | Structural Typology | Functional Domain | Technical Description |
| :--- | :--- | :--- | :--- |
| **`InvoiceNo`** | String / Object | Transaction Identifier | A unique 6-digit structural string identifier assigned to each transaction. Invoices starting with "C" denote formal system cancellations. |
| **`StockCode`** | String / Object | Operations / Catalog | A unique alphanumeric variable assigned to each separate product in the inventory ledger. |
| **`Description`** | String / Object | Product Inventory | The explicit text name corresponding to each item listed in the inventory database. |
| **`Quantity`** | Integer | Order Dimension | The exact number of product units purchased per transaction line-item. Negative numbers signify ledger corrections or returns. |
| **`InvoiceDate`** | Datetime | Temporal Anchor | The exact timestamp marking when a transaction was generated in the system database. |
| **`UnitPrice`** | Float | Financial Variable | Product price per individual unit expressed in Great Britain Pounds (£). |
| **`CustomerID`** | Float / String | Demographic Key | A unique 5-digit nominal identifier systematically tracking specific user accounts. |
| **`Country`** | String / Object | Spatial / Geo-Location | The country where the purchasing customer is registered. |

---

## 🔍 Data Cleaning & Quality Assurance Metrics
Raw real-world transactional data contains structural inconsistencies. To protect model training from skewed variances, rigorous data cleaning was executed across specific functional metrics:

1. **Handling Missing Identifiers (`CustomerID`):** * **Issue:** 135,080 records (24.93% of the raw dataset) lacked a `CustomerID`.
   * **Action:** Because unsupervised customer-level behavioral clustering requires unique identities, rows without a valid `CustomerID` were dropped.
2. **Filtering System Cancellations & Bad Stock Operations:**
   * **Issue:** Transactions with an `InvoiceNo` starting with `"C"` represent formal order cancellations, generating negative values in the `Quantity` column. Additionally, internal stock adjustments created zero or negative values in `UnitPrice`.
   * **Action:** Outlined a strict subsetting logic to isolate active, completed transactions (`Quantity > 0` and `UnitPrice > 0`). This removed internal bookkeeping entries and data noise.
3. **Removing Duplicate Rows:**
   * **Action:** Dropped 5,268 explicit duplicate entries across the transaction matrix to prevent artificial inflation of customer transaction frequencies.

---

## 📈 Exploratory Data Analysis (EDA) & Core Findings
Exploratory Data Analysis provided critical insights into underlying transaction distributions, geographic variations, and revenue-generating mechanics.

### 1. Global Revenue & Geographic Dispersion
* **Total Gross Sales:** The platform generated **£10,666,684.54** across the active transactional pipeline.
* **Geographic Concentration:** The United Kingdom dominates transaction volume. However, international markets contribute significantly to revenue. When evaluating revenue performance outside the UK, countries like the **Netherlands, EIRE (Ireland), Germany, and France** emerge as key growth drivers.

### 2. Temporal Revenue Fluctuations & Seasonality
* Tracking monthly revenue trends revealed strong seasonal purchasing behaviors.
* Sales remain stable through the first three quarters of the year before rising sharply in the fourth quarter, peaking at over **£1.5M in November 2011** due to pre-holiday inventory restocking.

### 3. Pricing and Volume Distributions
* **Quantity and UnitPrice Skewness:** Both metrics display heavily right-skewed Pareto distributions. Most line items consist of low-cost gift novelties (median item price under £3.00) bought in small quantities, punctuated by occasional high-volume wholesale transactions.

---

## 🛠️ Feature Engineering: The RFM Framework
To transform transaction logs into input data suitable for clustering algorithms, individual records were aggregated into user-level operational vectors using an **RFM (Recency, Frequency, Monetary) Model**.

### RFM Feature Formulas & Definitions
1. **Recency ($R_i$):** Measures behavioral detachment. Computed as the number of days between a customer's last purchase date and the maximum snapshot date in the database.
   $$R_i = \max(\text{InvoiceDate}) - \max(\text{InvoiceDate}_i)$$
2. **Frequency ($F_i$):** Measures platform interaction density. Computed as the total number of unique invoices generated by a specific customer account.
   $$F_i = \text{Count}(\text{Unique InvoiceNo}_i)$$
3. **Monetary Value ($M_i$):** Measures financial scale. Computed as the sum of net spend across all line-items linked to a customer account.
   $$M_i = \sum (\text{Quantity} \times \text{UnitPrice})$$

### Advanced Preprocessing & Mathematical Scaling
* **Outlier Capping via Interquartile Range (IQR):** Extreme wholesale transactions can distort cluster centroids during model training. To prevent this, data points beyond the **99th percentile** were capped using IQR clipping, protecting model stability while preserving core data features.
* **Standard Scaling (Z-Score Normalization):** K-Means relies heavily on Euclidean distance calculations. Because feature dimensions have vastly different units (days vs. transaction counts vs. cash value), all features were standard scaled to ensure equal weight during clustering:
  $$\mathbf{z} = \frac{\mathbf{x} - \mu}{\sigma}$$

---

## 🤖 K-Means Clustering Theory & Optimization
The core segmentation engine utilizes the **K-Means Clustering algorithm**, an efficient and scalable method for partitioning multi-dimensional numeric datasets.

### Mathematical Objective Function
The algorithm iteratively divides the user base into distinct clusters by minimizing the within-cluster sum of squares (WCSS), driving cluster centroids toward local convergence:
$$J = \sum_{j=1}^{K} \sum_{i=1}^{n} \| x_i^{(j)} - \mu_j \|^2$$
Where $K$ represents the total number of clusters, $n$ is the number of customer samples, $x_i^{(j)}$ is a specific feature vector, and $\mu_j$ denotes the centroid of cluster $j$.

### Determining the Optimal Cluster Value ($K$)
To evaluate cluster separation and find the optimal balance between detail and simplicity, two validation techniques were applied:

1. **The Elbow Method (WCSS Evaluation):** WCSS was plotted for values from $K = 1$ to $K = 10$. The line showed a distinct inflection point or "elbow" at **$K = 5$**, indicating diminishing returns in variance explanation beyond this point.
2. **Silhouette Coefficient Validation:** The Silhouette Score measures how similar an instance is to its own cluster compared to neighboring clusters. Evaluating these scores confirmed that **$K = 5$** provides distinct, well-separated cluster boundaries.

---

## 📉 Dimensionality Reduction & Manifold Visualization
To validate and interpret the multi-dimensional RFM customer clusters, two dimensionality reduction techniques were used to map the feature space into two dimensions.

### 1. Principal Component Analysis (PCA)
PCA was used as a linear transformation technique to project features into orthogonal axes representing maximum variance. This simplifies the high-dimensional data, confirming clear geometric boundaries and minimal overlap between the clusters.

### 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)
To complement PCA, t-SNE was applied as a non-linear dimensionality reduction technique. By mapping high-dimensional proximities into low-dimensional conditional probabilities, t-SNE isolates fine-grained local patterns, showing distinct, well-separated cluster regions across the customer base.

---

## 👥 Customer Segment Analysis & Behavioral Archetypes
The trained K-Means pipeline identified five distinct customer segments based on their explicit purchasing habits, interaction frequencies, and monetary scale.

### 🥇 Segment 1: Champions & Premium Wholesalers
* **Behavioral Dynamics:** Exceptionally high frequency and high monetary spend with low recency values. These are frequent, high-volume buyers.
* **Economic Footprint:** High total spend, large basket sizes, and highly consistent purchase intervals.
* **Strategic Classification:** Core high-value assets driving a significant portion of structural platform revenue.

### 🥈 Segment 2: Loyal High-Value VIPs
* **Behavioral Dynamics:** Consistent, high-frequency shopping habits combined with solid monetary value.
* **Economic Footprint:** Regular platform engagement with above-average total spend across transactions.
* **Strategic Classification:** Highly dependable customer group that provides steady revenue support.

### 🥉 Segment 3: Consistent Mid-Market Buyers
* **Behavioral Dynamics:** Predictable purchasing patterns with average transaction frequencies and moderate spend.
* **Economic Footprint:** Stable transaction intervals and mid-tier basket sizes.
* **Strategic Classification:** Represents a reliable baseline of recurring revenue with clear opportunities for incremental value growth.

### ⚠️ Segment 4: At-Risk / Dormant Accounts
* **Behavioral Dynamics:** High recency values (extended periods of inactivity) combined with low purchase frequencies and minimal spend.
* **Economic Footprint:** Formerly active accounts showing clear signs of declining user engagement.
* **Strategic Classification:** Primary targets for automated win-back marketing campaigns to prevent structural customer churn.

### 📉 Segment 5: Low-Engagement / Occasional Budget Buyers
* **Behavioral Dynamics:** Low frequency, small spending footprints, and extended intervals between active orders.
* **Economic Footprint:** Small transaction amounts across infrequent platform interactions.
* **Strategic Classification:** Low-yield customer accounts; best suited for automated, low-cost marketing approaches.

---

## 🚀 Tailored Marketing & Business Growth Strategies
To optimize marketing spend, personalized retention and acquisition strategies were developed for each customer archetype:

* **Champions & Premium Wholesalers:**
  * Implement premium loyalty tiers, early access to new product rollouts, and dedicated account support.
  * Provide value-based bulk volume discounts and priority order logistics to secure long-term partnerships.
* **Loyal High-Value VIPs:**
  * Deploy personalized product cross-selling strategies based on historical purchase trends.
  * Offer milestone-based rewards and exclusive referral incentives to keep engagement high.
* **Consistent Mid-Market Buyers:**
  * Introduce value-driven product bundles and threshold-based free shipping offers to increase average order values (AOV).
  * Use personalized email updates highlighting items related to their frequent categories.
* **At-Risk / Dormant Accounts:**
  * Launch automated "We Miss You" email campaigns featuring tailored reactivating discount vouchers.
  * Conduct simple feedback surveys to identify and address common pain points in the user journey.
* **Low-Engagement / Occasional Budget Buyers:**
  * Include in automated, low-cost marketing outreach for major seasonal clear-out promotions and sitewide holiday sales.

---

## 🎨 System Visualizations Gallery
The underlying notebook generates a series of plots to assess cluster quality and explore the transaction data:

<p align="center">
  <img src="images/cluster_visualization.png" width="750" alt="Dimensional Projections of K-Means Clusters">
</p>
<p align="center"><em>Figure: Behavioral separations visualized using low-dimensional mapping layers.</em></p>

---

## 🔮 Future Architecture Enhancements
* **Advanced Clustering Implementations:** Test density-based and hierarchical algorithms like DBSCAN and Agglomerative Clustering to capture complex, non-spherical customer groupings.
* **Streaming Analytical Infrastructure:** Integrate streaming tools like Apache Kafka or PySpark to update customer profile clusters in real time based on active browsing sessions.
* **Predictive Lifetime Value (LTV) Modeling:** Combine current behavioral clusters with supervised survival analysis models to forecast individual Customer Lifetime Value directly from initial transaction inputs.
