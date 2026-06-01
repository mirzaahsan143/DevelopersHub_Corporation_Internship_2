# 🏪 Global Superstore — Business Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

> An interactive, production-ready Business Intelligence dashboard for exploring sales, profit, customer, and segment performance of a global retail company — built with Streamlit and Plotly.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [EDA Highlights](#eda-highlights)
- [Dashboard Features](#dashboard-features)
- [Business Insights](#business-insights)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 🔍 Project Overview

This project implements a complete end-to-end Business Intelligence solution for **Global Superstore**, a worldwide retail company operating across 147 countries and 7 global markets. The solution includes:

- A comprehensive **Jupyter Notebook** with data cleaning, EDA, and visualizations
- A production-ready **Streamlit dashboard** with interactive filters and KPI cards
- Automated **data storytelling** that generates real business insights from filtered data

---

## 🧩 Problem Statement

Global Superstore generates millions in revenue annually but lacked a centralized analytical system to:

- Monitor KPIs across regions, categories, and customer segments in real-time
- Identify underperforming product lines and high-value customers
- Understand the profitability impact of discount policies
- Support evidence-based management decisions across all organizational levels

A business intelligence dashboard resolves this by transforming 51,000+ raw transactions into actionable visual intelligence.

---

## 🎯 Objectives

1. Clean and validate the Global Superstore dataset for analysis-ready state
2. Conduct thorough EDA across sales, profit, customers, segments, and geography
3. Create professional visualizations with business interpretation
4. Build a fully interactive Streamlit BI dashboard with dynamic sidebar filters
5. Generate automated, data-driven business insights from filtered selections
6. Deliver actionable recommendations for management

---

## 📊 Dataset Description

| Attribute | Detail |
|-----------|--------|
| **Source** | Global Superstore (Kaggle / Tableau Sample) |
| **Records** | 51,290 rows |
| **Features** | 24 columns |
| **Time Period** | Jan 2011 – Dec 2014 |
| **Markets** | 7 global markets (US, EU, APAC, LATAM, EMEA, Africa, Canada) |
| **Regions** | 13 sub-regions |
| **Customers** | 795 unique customers |
| **Categories** | 3 (Technology, Furniture, Office Supplies) |
| **Sub-Categories** | 17 |

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| Order Date | Date | Purchase date |
| Customer Name | String | Customer identifier |
| Segment | Category | Consumer / Corporate / Home Office |
| Region / Market | Category | Geographic breakdown |
| Category / Sub-Category | Category | Product classification |
| Sales | Float | Revenue per line item |
| Profit | Float | Net profit per line item |
| Discount | Float | Discount rate (0–1) |
| Shipping Cost | Float | Logistics cost |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical visualizations |
| **Plotly** | Interactive charts in dashboard |
| **Streamlit** | Dashboard web application framework |
| **xlrd / openpyxl** | Excel file reading |
| **Jupyter Notebook** | EDA and documentation |

---

## 📁 Project Structure

```
GlobalSuperstore/
│
├── app.py                  # Streamlit dashboard (main application)
├── notebook.ipynb          # Complete EDA & analysis notebook
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── dataset/
│   └── Global_Superstore.xls   # Source dataset
│
└── assets/                 # Optional: screenshots, images
```

---

## 📈 EDA Highlights

### Sales Analysis
- Total sales of **$12.64M** over four years with consistent YoY growth
- **Q4 seasonality** is prominent — October to December generates peak revenue every year
- **APAC** and **EU** are the dominant global markets

### Profit Analysis
- Overall profit margin of **11.6%** ($1.47M on $12.64M revenue)
- **Tables** and **Bookcases** sub-categories are chronically loss-making
- **Discounts above 30%** consistently result in negative profit per order

### Customer Analysis
- **795 unique customers** with a Pareto distribution — top ~20% drive ~80% of revenue
- **Tom Ashbrook** is the highest-value customer with $40,488 in lifetime purchases
- Low customer concentration risk overall

### Segment Analysis
- **Consumer** segment: 51% of sales — the dominant segment
- **Home Office**: smallest but highest profit margin per order
- **Corporate**: steady growth, second-largest contributor

---

## 🖥️ Dashboard Features

### Sidebar Filters
- **Year** multi-select (2011–2014)
- **Region** multi-select (13 regions)
- **Category** multi-select (3 categories)
- **Sub-Category** cascading multi-select (updates based on selected categories)

### KPI Cards
- 💰 Total Sales
- 📈 Total Profit
- 🧾 Total Orders
- 👥 Unique Customers
- 🛒 Average Order Value
- 📊 Profit Margin %

### Interactive Charts (10 Charts)
1. Monthly Sales Trend (area + line)
2. Monthly Profit Trend (color-coded bar)
3. Yearly Sales & Profit (bar charts)
4. Sales by Region (horizontal bar)
5. Profit by Region (color-coded horizontal bar)
6. Sales & Profit by Category (pie + bar)
7. Sales & Profit by Sub-Category
8. Segment-wise Sales & Profit
9. Segment × Category Heatmap
10. Top 10 Customers by Sales & Profit
11. Top 10 Products by Sales
12. Sales vs Profit Bubble by Market
13. Discount Rate vs Profit (scatter)
14. Sales by Ship Mode (pie)

### Data Storytelling
Automatically generates 8 contextual business insights based on whatever the user has filtered — dynamically updated on every filter change.

### Raw Data Explorer
Collapsible section showing the top 500 rows of the filtered dataset.

---

## 💡 Business Insights

| # | Insight |
|---|---------|
| 1 | Technology drives the highest revenue (37% of total sales) |
| 2 | Consumer segment accounts for 51% of all sales |
| 3 | Discounts above 30% result in negative average profit |
| 4 | Tables sub-category generates consistent losses — needs pricing review |
| 5 | APAC is the largest and most profitable global market |
| 6 | Top 10% of customers generate ~50% of total revenue |
| 7 | Q4 generates approximately 35% more revenue than Q1 |
| 8 | Standard Class is the most-used shipping mode (60%+ of orders) |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Total Revenue | $12.64M |
| Total Profit | $1.47M |
| Profit Margin | 11.6% |
| Total Orders | 25,035 |
| Unique Customers | 795 |
| Data Coverage | 4 years (2011–2014) |
| Countries Covered | 147 |

---

## 🚀 Running the Dashboard

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app
```bash
streamlit run app.py
```

### 3. Open notebook (optional)
```bash
jupyter notebook notebook.ipynb
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 🔮 Future Improvements

1. **Predictive Forecasting** — Add ARIMA/Prophet sales forecasting for next 12 months
2. **RFM Customer Segmentation** — Recency-Frequency-Monetary model for churn prediction
3. **Real-Time Data Pipeline** — Connect to live database (PostgreSQL / BigQuery)
4. **Geo-Map Visualization** — Choropleth map for country-level performance
5. **Export Functionality** — Allow users to download filtered data as CSV/Excel
6. **Email Alerts** — Automated weekly KPI digest for management
7. **Multi-User Authentication** — Role-based access with department-level filters
8. **Mobile Optimization** — Responsive layout for tablet/phone viewing

---

## 📄 License

This project is developed for educational and internship purposes.

---

*Built using Python, Streamlit, and Plotly*
