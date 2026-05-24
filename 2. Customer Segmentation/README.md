# 🛍️ Customer Segmentation Using Unsupervised Learning

**Data Science Internship Project – Task 2 | DevelopersHub Corporation**
*Developed by Mirza Muhammad Ahsan*

---

## 📌 Project Overview
In the highly competitive retail sector, understanding customer behavior is essential for targeted marketing. This project applies **unsupervised machine learning** to segment mall customers based on their purchasing habits and demographic data. By grouping customers into distinct clusters, the mall can deploy personalized marketing campaigns, optimize product placement, and ultimately boost customer retention and revenue.

---

## 🎯 Objectives
1. Perform **Exploratory Data Analysis (EDA)** to uncover trends in customer demographics.
2. Clean and preprocess the data for distance-based machine learning algorithms.
3. Determine the optimal number of customer segments using the **Elbow Method** and **Silhouette Score**.
4. Apply **K-Means Clustering** to categorize customers into distinct behavioral groups.
5. Utilize **PCA** and **t-SNE** to reduce dimensionality and visualize complex clusters.
6. Formulate actionable **marketing strategies** tailored to each specific customer segment.

---

## 🗂️ Dataset Description
The analysis utilizes the `Mall_Customers.csv` dataset, which contains 200 records and the following features:

| Feature | Description |
| :--- | :--- |
| **CustomerID** | Unique identifier for each customer (Dropped during modeling). |
| **Gender** | Male or Female. |
| **Age** | Age of the customer in years. |
| **Annual Income (k$)** | The customer's yearly income in thousands of dollars. |
| **Spending Score (1-100)** | A metric assigned by the mall based on customer spending behavior (higher score = higher spending). |

---

## ⚙️ Methodology & Process

### 1. Data Cleaning & Preprocessing
To ensure the clustering algorithms performed optimally, the following data cleaning and preparation steps were taken:
* **Missing Values & Duplicates:** Verified the dataset structural integrity. Found 0 missing values and 0 duplicate rows.
* **Feature Selection:** Dropped `CustomerID` as it provides no variance or behavioral insight for clustering.
* **Formatting:** Renamed columns (e.g., `Genre` to `Gender`, `Annual Income (k$)` to `Annual_Income`) for standardized coding practices.
* **Feature Scaling:** Applied `StandardScaler` to normalize numerical features (`Age`, `Annual_Income`, `Spending_Score`). This is a critical metric for K-Means, which relies on Euclidean distance, ensuring no single feature dominates the clustering due to its raw scale.

### 2. Exploratory Data Analysis (EDA) Findings
* **Demographics:** The customer base leans slightly female (56%). The average customer age is roughly 38.8 years.
* **Correlations:** A weak negative correlation exists between Age and Spending Score, indicating younger customers tend to spend more.
* **Visual Patterns:** Scatter plots of Annual Income vs. Spending Score visually revealed roughly 5 natural groupings of customers even before machine learning was applied.

### 3. Machine Learning (K-Means Clustering)
* **Finding Optimal Clusters:** Evaluated cluster counts using the **Elbow Method** (WCSS) and **Silhouette Scores**. Both metrics confirmed that **K=5** was the optimal number of clusters for this dataset.
* **Modeling:** Trained K-Means models on 2D space (Income + Spending) for highly interpretable business results, and 3D space (Age + Income + Spending) for deeper analysis.
* **Dimensionality Reduction:** Applied **Principal Component Analysis (PCA)** and **t-SNE** to map the multi-dimensional feature space into a 2D plane, validating that the 5 clusters were tight and well-separated.

---

## 📊 Key Findings: The 5 Customer Segments

The K-Means algorithm successfully segmented the customers into 5 distinct profiles:

1.  🎯 **Cluster 0: Sensible Savers** * **Profile:** Middle-income, moderate to low spenders.
    * **Strategy:** Focus on value bundles, loyalty points, and cashback offers to gradually increase their visit frequency.
2.  💎 **Cluster 1: Premium Customers** * **Profile:** High-income, high spenders. The most valuable demographic.
    * **Strategy:** Offer VIP memberships, exclusive preview events, and personalized high-end luxury experiences to maintain loyalty.
3.  🛒 **Cluster 2: Budget Shoppers** * **Profile:** Low-income, low spenders.
    * **Strategy:** Target with discount coupons, clearance sales, and daily deals to reduce purchase friction.
4.  ⭐ **Cluster 3: Careless Spenders** * **Profile:** Low-income, high spenders (impulsive buyers).
    * **Strategy:** Implement Buy-Now-Pay-Later (BNPL) options, limited-edition drops, and influencer-driven campaigns.
5.  💰 **Cluster 4: Conservative Elites** * **Profile:** High-income, low spenders. A major untapped opportunity.
    * **Strategy:** Avoid aggressive discount marketing; instead, offer premium consultation services (like style advisors) and exclusive, high-value experiences.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (K-Means, PCA, t-SNE, StandardScaler, Silhouette metrics)

---

## 🚀 How to Run the Project
1. Clone this repository to your local machine.
2. Ensure you have Jupyter Notebook installed, along with the necessary libraries (`pip install pandas numpy matplotlib seaborn scikit-learn`).
3. Place `Mall_Customers.csv` in the same directory as the notebook.
4. Open `Customer_Segmentation_Project.ipynb` and run the cells sequentially to reproduce the analysis and visualizations.

---
*This repository serves as a portfolio project demonstrating proficiency in data wrangling, exploratory data analysis, and unsupervised machine learning techniques.*
