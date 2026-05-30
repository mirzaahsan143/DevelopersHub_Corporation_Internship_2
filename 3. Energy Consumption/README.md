<div align="center">
  <h1>⚡ Energy Consumption Time Series Forecasting</h1>
  <h3>Predictive Analytics & Data Science</h3>
</div>

<hr>

## 📑 Project Overview

This project focuses on short-term univariate and multivariate time series forecasting of household power consumption. Accurate energy forecasting is critical for modern grid stability, enabling utility companies to optimize real-time supply and demand. By moving beyond raw meter data, this analysis extracts actionable temporal patterns to predict future electricity consumption. The quantitative insights derived from this forecasting pipeline are designed to empower smart home automation, optimize demand-response utility programs, and isolate load anomalies (e.g., energy theft or appliance degradation) when real-time consumption deviates by **>2σ** from forecasted expectations. Households utilizing these predictive insights on dynamic tariffs can achieve estimated annual cost savings of **8–15%**.

## 🗄️ Dataset Description

The analysis leverages the **UCI Household Power Consumption** dataset, featuring minute-resolution telemetry from a single household in Sceaux, France, spanning a six-month period (January–June 2007). 

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr style="background-color: #f8f9fa;">
      <th>Feature Group</th>
      <th>Variables</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Target</b></td>
      <td><code>Global_active_power</code></td>
      <td>Total active power drawn by the household (measured in kilowatts).</td>
    </tr>
    <tr>
      <td><b>Electrical Metrics</b></td>
      <td><code>Global_reactive_power</code>, <code>Voltage</code>, <code>Global_intensity</code></td>
      <td>Granular electrical properties including reactive power (kW), average voltage (V), and current intensity (A).</td>
    </tr>
    <tr>
      <td><b>Sub-metering</b></td>
      <td><code>Sub_metering_1</code>, <code>Sub_metering_2</code>, <code>Sub_metering_3</code></td>
      <td>Watt-hour (Wh) energy consumption split across specific circuits: Kitchen (1), Laundry (2), and Water Heater / AC (3).</td>
    </tr>
  </tbody>
</table>
<br>
<i>* The raw dataset contains <b>260,640 records</b> and 10 initial columns.</i>

## 🧹 Data Cleaning & Preprocessing

Real-world sensor telemetry is highly susceptible to missing values and noise. The dataset was standardized through rigorous, temporally-aware preprocessing:

* **Temporal Alignment:** <code>Date</code> and <code>Time</code> columns were concatenated and converted into a continuous <code>Datetime</code> index to validate the minute-by-minute temporal grid.
* **Missing Value Imputation:** <b>3,771 missing records (1.45%)</b> were identified. To preserve local temporal dependencies and avoid artificial smoothing, gaps of ≤60 minutes were processed using forward-fill (<code>ffill</code>), while any remaining trailing gaps utilized backward-fill (<code>bfill</code>).
* **Anomaly Rectification:** Impossible negative active-power sensor readings were isolated and clipped to a <b>0 kW</b> lower bound. 
* **Feature Engineering:**
  * <code>Sub_metering_total</code>: Aggregation of all three isolated circuits.
  * <code>Active_power_wh</code>: Transformation of the target variable from kW to Watt-hours (Wh) per minute.
  * <code>Energy_remaining</code>: Computed unmetered household consumption by subtracting <code>Sub_metering_total</code> from <code>Active_power_wh</code>.

## 📈 Exploratory Data Analysis

Initial statistical evaluation yielded several critical findings dictating the modeling architecture:

* **Target Variable Skewness:** The `Global_active_power` distribution is heavily right-skewed (Coefficient of Variation = **102.3%**). The massive concentration of readings sits between **0.2 kW and 1.5 kW** (overnight/baseline load), while a long tail stretches up to **10.67 kW** during peak usage. The Mean (**1.15 kW**) substantially outweighs the Median (**0.53 kW**).
* **Multicollinearity Discovery:** `Global_active_power` and `Global_intensity` exhibit near-perfect linear correlation (**r ≈ 0.99**), strictly adhering to Ohm's Law.
* **Voltage Depression:** A definitive negative correlation was detected between `Voltage` and active power, confirming that localized heavy loads successfully depress line voltage.
* **Circuit Impact:** <code>Sub_metering_3</code> (Water Heater & AC) maintains the highest mean consumption (**5.74 Wh**) and largest standard deviation (**8.15 Wh**), proving to be the primary driver of household demand spikes.

## 🧠 Analysis & Modeling

The predictive architecture evaluates three distinct algorithmic approaches to handle the temporal dependence, multi-seasonality, and non-stationarity inherent in energy telemetry.

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr style="background-color: #f8f9fa;">
      <th width="15%">Model</th>
      <th width="20%">Algorithmic Architecture</th>
      <th width="65%">Quantitative Findings & Performance Evaluation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>ARIMA</b></td>
      <td>Auto-Regressive Integrated Moving Average</td>
      <td>Applied as a univarate statistical baseline. Augmented Dickey-Fuller (ADF) testing on daily aggregates returned a Test Statistic of <b>-2.8048</b> (p-value: <b>0.0576</b>). This narrowly failed the 5% critical value (-2.878), confirming non-stationarity and necessitating first-order differencing (<b>d=1</b>) to stabilize the mean.</td>
    </tr>
    <tr>
      <td><b>Facebook Prophet</b></td>
      <td>Generalized Additive Model (GAM)</td>
      <td>Highly effective at mapping general directionality, weekly seasonality, and broad cyclic trends. However, Prophet systematically under-predicted the magnitude of extreme, localized evening consumption spikes due to its reliance on linear/logistic growth smoothing.</td>
    </tr>
    <tr>
      <td><b>XGBoost</b></td>
      <td>Gradient Boosted Decision Trees</td>
      <td>Delivered the most robust handling of the skewed active power distribution. The tree-based architecture successfully modeled high-variance peak loads without requiring target log-transformations. To prevent overfitting to sensor noise, the estimator was heavily regularized with <code>learning_rate = 0.05</code>, <code>max_depth = 6</code>, <code>min_child_weight = 5</code>, and an 80% feature/row subsampling ratio.</td>
    </tr>
  </tbody>
</table>
