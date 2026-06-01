"""
Global Superstore Business Intelligence Dashboard
Author: Data Science Team
Description: Interactive Streamlit dashboard for analyzing Sales, Profit,
             Customer, and Segment performance of Global Superstore.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Global Superstore BI Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
        /* Global font */
        html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

        /* KPI card container */
        .kpi-card {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
            border-radius: 12px;
            padding: 20px 24px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            margin-bottom: 12px;
        }
        .kpi-card .kpi-title {
            font-size: 13px;
            letter-spacing: 1px;
            text-transform: uppercase;
            opacity: 0.85;
            margin-bottom: 8px;
        }
        .kpi-card .kpi-value {
            font-size: 28px;
            font-weight: 700;
        }
        .kpi-card .kpi-icon {
            font-size: 20px;
            margin-bottom: 6px;
        }

        /* Header */
        .main-header {
            background: linear-gradient(135deg, #0d1b2a 0%, #1b4f72 100%);
            color: white;
            padding: 22px 30px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .main-header h1 { margin: 0; font-size: 30px; }
        .main-header p { margin: 4px 0 0; font-size: 14px; opacity: 0.8; }

        /* Section title */
        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: #1b4f72;
            border-left: 5px solid #2d6a9f;
            padding-left: 12px;
            margin: 24px 0 14px;
        }

        /* Insight box */
        .insight-box {
            background: #f0f7ff;
            border: 1px solid #b3d4f5;
            border-left: 5px solid #2d6a9f;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 10px;
            font-size: 14px;
            color: #1a3a5c;
        }
        .insight-box span { font-weight: 700; color: #1b4f72; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #0d1b2a;
        }
        [data-testid="stSidebar"] * { color: white !important; }

        /* Chart borders */
        .plot-container { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and preprocess the Global Superstore dataset."""
    # Support both xls (in dataset folder) and csv (if user converts it)
    base = os.path.dirname(os.path.abspath(__file__))
    xls_path = os.path.join(base, "dataset", "Global_Superstore.xls")
    csv_path = os.path.join(base, "dataset", "Global_Superstore.csv")

    if os.path.exists(xls_path):
        df = pd.read_excel(xls_path, engine="xlrd")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="latin-1")
    else:
        st.error("Dataset not found. Place Global_Superstore.xls in the dataset/ folder.")
        st.stop()

    # ── Date formatting ──────────────────────────────────────────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"])

    # ── Derived time columns ─────────────────────────────────────────────
    df["Year"]       = df["Order Date"].dt.year
    df["Month"]      = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%b")
    df["YearMonth"]  = df["Order Date"].dt.to_period("M").astype(str)

    # ── Drop Postal Code (mostly null, not used in analysis) ─────────────
    df.drop(columns=["Postal Code"], inplace=True, errors="ignore")

    # ── Remove any exact duplicate rows ─────────────────────────────────
    df.drop_duplicates(inplace=True)

    return df


df_raw = load_data()


# ─────────────────────────────────────────────
# SIDEBAR  —  FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏪 Global Superstore")
    st.markdown("---")
    st.markdown("### 🎛️ Dashboard Filters")

    # Year filter
    all_years = sorted(df_raw["Year"].unique())
    selected_years = st.multiselect(
        "📅 Year",
        options=all_years,
        default=all_years,
    )

    # Region filter
    all_regions = sorted(df_raw["Region"].unique())
    selected_regions = st.multiselect(
        "🌍 Region",
        options=all_regions,
        default=all_regions,
    )

    # Category filter
    all_cats = sorted(df_raw["Category"].unique())
    selected_cats = st.multiselect(
        "📦 Category",
        options=all_cats,
        default=all_cats,
    )

    # Sub-Category filter  (cascades from Category)
    sub_cat_options = sorted(
        df_raw[df_raw["Category"].isin(selected_cats)]["Sub-Category"].unique()
    )
    selected_subcats = st.multiselect(
        "🏷️ Sub-Category",
        options=sub_cat_options,
        default=sub_cat_options,
    )

    st.markdown("---")
    st.markdown("#### 📊 About")
    st.markdown(
        "Global Superstore BI Dashboard  \n"
        "51K+ orders · 2011-2014  \n"
        "Built with Streamlit & Plotly"
    )


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_raw[
    df_raw["Year"].isin(selected_years) &
    df_raw["Region"].isin(selected_regions) &
    df_raw["Category"].isin(selected_cats) &
    df_raw["Sub-Category"].isin(selected_subcats)
].copy()

if df.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust the sidebar selections.")
    st.stop()


# ─────────────────────────────────────────────
# PLOTLY COLOR PALETTE
# ─────────────────────────────────────────────
PALETTE   = px.colors.qualitative.Bold
BLUE_SEQ  = px.colors.sequential.Blues
GREEN_SEQ = px.colors.sequential.Greens
RED_SEQ   = px.colors.sequential.Reds

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Segoe UI", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)


def apply_layout(fig, title=""):
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(size=15, color="#1b4f72")))
    return fig


# ─────────────────────────────────────────────
# DASHBOARD HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏪 Global Superstore · Business Intelligence Dashboard</h1>
    <p>Interactive analysis of Sales · Profit · Customers · Segments &nbsp;|&nbsp; Data: 2011–2014</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
total_sales    = df["Sales"].sum()
total_profit   = df["Profit"].sum()
num_orders     = df["Order ID"].nunique()
num_customers  = df["Customer Name"].nunique()
avg_order_val  = df.groupby("Order ID")["Sales"].sum().mean()
profit_margin  = (total_profit / total_sales * 100) if total_sales else 0

def kpi_card(icon, title, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>"""

st.markdown('<div class="section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(kpi_card("💰", "Total Sales", f"${total_sales:,.0f}"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("📈", "Total Profit", f"${total_profit:,.0f}"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("🧾", "Total Orders", f"{num_orders:,}"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("👥", "Customers", f"{num_customers:,}"), unsafe_allow_html=True)
with c5:
    st.markdown(kpi_card("🛒", "Avg Order Value", f"${avg_order_val:,.0f}"), unsafe_allow_html=True)
with c6:
    st.markdown(kpi_card("📊", "Profit Margin", f"{profit_margin:.1f}%"), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — SALES & PROFIT TRENDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📅 Sales & Profit Trends Over Time</div>', unsafe_allow_html=True)

monthly = (
    df.groupby("YearMonth")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("YearMonth")
)

col1, col2 = st.columns(2)

with col1:
    fig = px.line(monthly, x="YearMonth", y="Sales", markers=True,
                  color_discrete_sequence=["#2d6a9f"],
                  labels={"YearMonth": "Month", "Sales": "Sales ($)"})
    fig.update_traces(fill="tozeroy", fillcolor="rgba(45,106,159,0.1)", line_width=2)
    apply_layout(fig, "📈 Monthly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in monthly["Profit"]]
    fig = go.Figure(go.Bar(
        x=monthly["YearMonth"], y=monthly["Profit"],
        marker_color=colors,
        hovertemplate="Month: %{x}<br>Profit: $%{y:,.0f}<extra></extra>",
    ))
    apply_layout(fig, "📊 Monthly Profit Trend")
    st.plotly_chart(fig, use_container_width=True)

# Yearly
yearly = df.groupby("Year")[["Sales", "Profit"]].sum().reset_index()

col3, col4 = st.columns(2)
with col3:
    fig = px.bar(yearly, x="Year", y="Sales",
                 color_discrete_sequence=["#2d6a9f"],
                 text_auto=".2s",
                 labels={"Sales": "Sales ($)"})
    apply_layout(fig, "📅 Yearly Sales")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.bar(yearly, x="Year", y="Profit",
                 color="Profit",
                 color_continuous_scale=["#e74c3c", "#f9e79f", "#27ae60"],
                 text_auto=".2s",
                 labels={"Profit": "Profit ($)"})
    apply_layout(fig, "📅 Yearly Profit")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 2 — REGIONAL ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🌍 Regional Performance</div>', unsafe_allow_html=True)

regional = (
    df.groupby("Region")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
)
regional["Profit Margin %"] = (regional["Profit"] / regional["Sales"] * 100).round(1)

col5, col6 = st.columns(2)

with col5:
    fig = px.bar(regional, x="Sales", y="Region", orientation="h",
                 color="Sales", color_continuous_scale=BLUE_SEQ,
                 text_auto=".2s", labels={"Sales": "Sales ($)"})
    apply_layout(fig, "🌍 Sales by Region")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

with col6:
    bar_colors = ["#e74c3c" if v < 0 else "#27ae60" for v in regional["Profit"]]
    fig = px.bar(regional, x="Profit", y="Region", orientation="h",
                 color_discrete_sequence=["#27ae60"],
                 text_auto=".2s", labels={"Profit": "Profit ($)"})
    fig.update_traces(marker_color=bar_colors)
    apply_layout(fig, "💹 Profit by Region")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 3 — CATEGORY ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📦 Category & Sub-Category Analysis</div>', unsafe_allow_html=True)

cat_df = (
    df.groupby("Category")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
)

subcat_df = (
    df.groupby("Sub-Category")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
)

col7, col8 = st.columns(2)

with col7:
    fig = px.pie(cat_df, names="Category", values="Sales",
                 color_discrete_sequence=PALETTE,
                 hole=0.45)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    apply_layout(fig, "🥧 Sales Share by Category")
    st.plotly_chart(fig, use_container_width=True)

with col8:
    fig = px.pie(cat_df, names="Category", values="Profit",
                 color_discrete_sequence=PALETTE,
                 hole=0.45)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    apply_layout(fig, "🥧 Profit Share by Category")
    st.plotly_chart(fig, use_container_width=True)

col9, col10 = st.columns(2)

with col9:
    fig = px.bar(subcat_df, x="Sub-Category", y="Sales",
                 color="Sales", color_continuous_scale=BLUE_SEQ,
                 text_auto=".2s")
    apply_layout(fig, "📊 Sales by Sub-Category")
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with col10:
    subcat_df_sorted = subcat_df.sort_values("Profit", ascending=False)
    profit_colors = ["#e74c3c" if v < 0 else "#27ae60"
                     for v in subcat_df_sorted["Profit"]]
    fig = px.bar(subcat_df_sorted, x="Sub-Category", y="Profit",
                 text_auto=".2s")
    fig.update_traces(marker_color=profit_colors)
    apply_layout(fig, "📊 Profit by Sub-Category")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 4 — SEGMENT ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">👥 Customer Segment Analysis</div>', unsafe_allow_html=True)

seg_df = (
    df.groupby("Segment")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
)

col11, col12 = st.columns(2)

with col11:
    fig = px.bar(seg_df, x="Segment", y="Sales",
                 color="Segment", color_discrete_sequence=PALETTE,
                 text_auto=".2s")
    apply_layout(fig, "👥 Sales by Segment")
    st.plotly_chart(fig, use_container_width=True)

with col12:
    fig = px.bar(seg_df, x="Segment", y="Profit",
                 color="Segment", color_discrete_sequence=PALETTE,
                 text_auto=".2s")
    apply_layout(fig, "👥 Profit by Segment")
    st.plotly_chart(fig, use_container_width=True)

# Segment x Category heatmap
seg_cat = (
    df.groupby(["Segment", "Category"])["Sales"]
    .sum()
    .reset_index()
    .pivot(index="Segment", columns="Category", values="Sales")
    .fillna(0)
)

fig = px.imshow(
    seg_cat, text_auto=".2s",
    color_continuous_scale=BLUE_SEQ,
    aspect="auto",
    labels=dict(x="Category", y="Segment", color="Sales ($)"),
)
apply_layout(fig, "🔲 Sales Heatmap: Segment × Category")
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 5 — CUSTOMER ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🏆 Top Customer Analysis</div>', unsafe_allow_html=True)

cust_sales = (
    df.groupby("Customer Name")["Sales"]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
    .head(10)
    .rename(columns={"Sales": "Total Sales"})
)

cust_profit = (
    df.groupby("Customer Name")["Profit"]
    .sum()
    .reset_index()
    .sort_values("Profit", ascending=False)
    .head(10)
    .rename(columns={"Profit": "Total Profit"})
)

col13, col14 = st.columns(2)

with col13:
    fig = px.bar(cust_sales, x="Total Sales", y="Customer Name",
                 orientation="h", color="Total Sales",
                 color_continuous_scale=BLUE_SEQ, text_auto=".2s")
    apply_layout(fig, "🏆 Top 10 Customers by Sales")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

with col14:
    fig = px.bar(cust_profit, x="Total Profit", y="Customer Name",
                 orientation="h", color="Total Profit",
                 color_continuous_scale=GREEN_SEQ, text_auto=".2s")
    apply_layout(fig, "🏆 Top 10 Customers by Profit")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 6 — PRODUCT ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🛍️ Top Products by Sales</div>', unsafe_allow_html=True)

top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
    .head(10)
)

fig = px.bar(top_products, x="Sales", y="Product Name",
             orientation="h", color="Sales",
             color_continuous_scale=BLUE_SEQ, text_auto=".2s",
             labels={"Sales": "Total Sales ($)"})
apply_layout(fig, "🛍️ Top 10 Products by Sales")
fig.update_coloraxes(showscale=False)
fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=420)
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 7 — MARKET ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🌐 Market-Level Performance</div>', unsafe_allow_html=True)

market_df = (
    df.groupby("Market")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
)
market_df["Profit Margin %"] = (market_df["Profit"] / market_df["Sales"] * 100).round(1)

col15, col16 = st.columns(2)

with col15:
    fig = px.bar(market_df, x="Market", y="Sales",
                 color="Sales", color_continuous_scale=BLUE_SEQ,
                 text_auto=".2s")
    apply_layout(fig, "🌐 Sales by Market")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col16:
    fig = px.scatter(market_df, x="Sales", y="Profit",
                     size="Sales", color="Market",
                     hover_name="Market",
                     color_discrete_sequence=PALETTE,
                     text="Market",
                     size_max=60)
    apply_layout(fig, "💹 Sales vs Profit Bubble Chart by Market")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 8 — DISCOUNT & SHIPPING ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🚢 Discount & Shipping Analysis</div>', unsafe_allow_html=True)

col17, col18 = st.columns(2)

with col17:
    # Discount effect on profit
    fig = px.scatter(df.sample(min(3000, len(df))),
                     x="Discount", y="Profit",
                     color="Category", opacity=0.5,
                     color_discrete_sequence=PALETTE,
                     trendline="lowess",
                     labels={"Discount": "Discount Rate", "Profit": "Profit ($)"})
    apply_layout(fig, "📉 Discount Rate vs Profit")
    st.plotly_chart(fig, use_container_width=True)

with col18:
    ship_df = (
        df.groupby("Ship Mode")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
    )
    fig = px.pie(ship_df, names="Ship Mode", values="Sales",
                 color_discrete_sequence=PALETTE, hole=0.4)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    apply_layout(fig, "🚢 Sales by Ship Mode")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 9 — DATA STORYTELLING / INSIGHTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">💡 Automated Business Insights</div>', unsafe_allow_html=True)

# Derive dynamic insights from the filtered dataset
top_region     = regional.iloc[0]["Region"]
top_region_s   = regional.iloc[0]["Sales"]
top_cat        = cat_df.iloc[0]["Category"]
top_cat_pct    = round(cat_df.iloc[0]["Sales"] / cat_df["Sales"].sum() * 100, 1)
top_seg        = seg_df.iloc[0]["Segment"]
top_seg_pct    = round(seg_df.iloc[0]["Sales"] / seg_df["Sales"].sum() * 100, 1)
top_cust       = cust_sales.iloc[0]["Customer Name"]
top_cust_s     = cust_sales.iloc[0]["Total Sales"]
worst_subcat   = subcat_df.sort_values("Profit").iloc[0]["Sub-Category"]
worst_profit   = subcat_df.sort_values("Profit").iloc[0]["Profit"]
best_margin_r  = regional.sort_values("Profit Margin %", ascending=False).iloc[0]

insights = [
    f"<span>🌍 Top Region:</span> <b>{top_region}</b> leads in sales with "
    f"<b>${top_region_s:,.0f}</b> in total revenue.",

    f"<span>📦 Category Leader:</span> <b>{top_cat}</b> accounts for "
    f"<b>{top_cat_pct}%</b> of total sales — the dominant revenue driver.",

    f"<span>👥 Segment Dominance:</span> The <b>{top_seg}</b> segment contributes "
    f"<b>{top_seg_pct}%</b> of sales, making it the primary customer base.",

    f"<span>🏆 Top Customer:</span> <b>{top_cust}</b> is the highest-value customer "
    f"with <b>${top_cust_s:,.0f}</b> in purchases.",

    f"<span>⚠️ Loss-Making Sub-Category:</span> <b>{worst_subcat}</b> has negative "
    f"profit of <b>${worst_profit:,.0f}</b> — pricing or cost review is recommended.",

    f"<span>📈 Best Profit Margin:</span> <b>{best_margin_r['Region']}</b> region has "
    f"the highest profit margin at <b>{best_margin_r['Profit Margin %']:.1f}%</b>.",

    f"<span>📊 Overall Margin:</span> The blended profit margin across all filtered "
    f"data is <b>{profit_margin:.1f}%</b>.",

    f"<span>📉 Discount Risk:</span> High discount rates (above 30%) strongly "
    f"correlate with negative profit — a pricing discipline review is advised.",
]

for insight in insights:
    st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 10 — RAW DATA EXPLORER
# ─────────────────────────────────────────────
with st.expander("🔍 Explore Filtered Raw Data"):
    cols_to_show = [
        "Order Date", "Order ID", "Customer Name", "Segment",
        "Region", "Category", "Sub-Category", "Product Name",
        "Sales", "Profit", "Quantity", "Discount", "Ship Mode"
    ]
    display_cols = [c for c in cols_to_show if c in df.columns]
    st.dataframe(
        df[display_cols]
        .sort_values("Order Date", ascending=False)
        .head(500)
        .reset_index(drop=True),
        use_container_width=True,
        height=350,
    )
    st.caption(f"Showing top 500 of {len(df):,} filtered records.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color: #7f8c8d; font-size: 13px;'>"
    "🏪 Global Superstore BI Dashboard &nbsp;|&nbsp; "
    "Built with Streamlit & Plotly &nbsp;|&nbsp; "
    "Data: 2011–2014"
    "</center>",
    unsafe_allow_html=True
)
