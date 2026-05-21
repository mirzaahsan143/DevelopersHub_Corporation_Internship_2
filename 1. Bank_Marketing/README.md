<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Term Deposit Subscription Prediction - README</title>
    <style>
        :root {
            --primary-color: #2b4c7e;
            --secondary-color: #2ecc71;
            --dark-color: #2c3e50;
            --light-color: #f8f9fa;
            --border-color: #e2e8f0;
            --text-muted: #718096;
            --accent-color: #e74c3c;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            background-color: #f4f6f9;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1100px;
            margin: 40px auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }

        .header {
            text-align: center;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 30px;
            margin-bottom: 40px;
        }

        .header h1 {
            color: var(--primary-color);
            margin: 0 0 10px 0;
            font-size: 2.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        .header h3 {
            color: var(--text-muted);
            font-weight: 400;
            margin: 0;
            font-size: 1.2rem;
        }

        .project-meta {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
            background: var(--light-color);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .meta-item {
            text-align: center;
        }

        .meta-item strong {
            display: block;
            color: var(--primary-color);
            font-size: 0.9rem;
            text-transform: uppercase;
            margin-bottom: 5px;
        }

        .meta-item span {
            font-size: 1.1rem;
            font-weight: 600;
        }

        h2 {
            color: var(--primary-color);
            border-left: 5px solid var(--primary-color);
            padding-left: 15px;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.75rem;
        }

        h3 {
            color: var(--dark-color);
            margin-top: 25px;
            font-size: 1.3rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 5px;
        }

        p {
            margin-bottom: 20px;
            text-align: justify;
        }

        ul, ol {
            margin-bottom: 20px;
            padding-left: 25px;
        }

        li {
            margin-bottom: 8px;
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #fff;
        }

        .badge-success { background-color: var(--secondary-color); }
        .badge-danger { background-color: var(--accent-color); }
        .badge-primary { background-color: var(--primary-color); }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95rem;
        }

        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
        }

        tr:hover {
            background-color: rgba(43, 76, 126, 0.02);
        }

        .highlight-box {
            background-color: #ebf4ff;
            border-left: 4px solid #3182ce;
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin: 25px 0;
        }

        .insight-box {
            background-color: #f0fff4;
            border-left: 4px solid var(--secondary-color);
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin: 25px 0;
        }

        .code-block {
            background-color: #1a202c;
            color: #edf2f7;
            padding: 15px 20px;
            border-radius: 6px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            margin: 20px 0;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 25px;
        }

        footer {
            text-align: center;
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>🏦 Term Deposit Subscription Prediction</h1>
        <h3>Bank Marketing Campaigns — Data Science Internship Project</h3>
        
        <div class="project-meta">
            <div class="meta-item">
                <strong>Dataset Source</strong>
                <span>UCI Repository</span>
            </div>
            <div class="meta-item">
                <strong>Records Count</strong>
                <span>45,211 rows</span>
            </div>
            <div class="meta-item">
                <strong>Best Performing Model</strong>
                <span>Random Forest</span>
            </div>
            <div class="meta-item">
                <strong>Top Predictor</strong>
                <span>Call Duration (`duration`)</span>
            </div>
            <div class="meta-item">
                <strong>Explainability Method</strong>
                <span>SHAP (TreeExplainer)</span>
            </div>
        </div>
    </div>

    <h2>📌 1. Project Overview & Problem Statement</h2>
    <p>
        Retail banking organizations frequently conduct direct telemarketing campaigns to encourage clients to subscribe to long-term financial products, specifically <strong>term deposits</strong>. A term deposit is a fixed-term investment where a customer locks away an amount of capital for a specified duration at a guaranteed interest rate. For banks, securing these deposits is vital because it provides stable, low-cost capital to fund lending activities and financial products.
    </p>
    <p>
        However, traditional phone-based marketing strategies suffer from immense inefficiency. Cold-calling or blindly contacting every customer inside a large database results in massive resource wastage (high telecom fees, agent fatigue, operational overhead) and causes customer annoyance, potentially damaging customer relations. 
    </p>
    <div class="highlight-box">
        <strong>The Core Challenge:</strong> Out of 45,211 customers contacted in this historic campaign dataset, only <strong>5,289 (11.70%)</strong> actually subscribed to the term deposit. The remaining 88.30% represented wasted effort.
    </div>
    <p>
        <strong>Objective:</strong> The primary objective of this project is to develop a robust binary classification pipeline that leverages machine learning to predict whether a customer will subscribe (<code class="badge badge-primary">yes</code> / 1) or refuse (<code class="badge badge-danger">no</code> / 0) a term deposit. By filtering leads and scoring customers beforehand, the marketing team can prioritize high-probability converts, maximize conversion rates, and minimize operational costs.
    </p>

    <h2>📦 2. Technical Stack & Libraries Used</h2>
    <p>
        The complete pipeline was developed using Python's extensive data science ecosystem. The primary components include:
    </p>
    <ul>
        <li><strong>Data Manipulation & Numerical Operations:</strong> <code>pandas</code> and <code>numpy</code> for reading data, slicing, dtypes inspection, handling tabular frames, and matrix formatting.</li>
        <li><strong>Data Visualization:</strong> <code>matplotlib.pyplot</code> and <code>seaborn</code> to build professional distributions, side-by-side counts, correlation matrices, and customized metric graphs.</li>
        <li><strong>Machine Learning Pipeline:</strong> <code>scikit-learn</code> for data preprocessing (<code>StandardScaler</code>), dataset splitting (<code>train_test_split</code>), classification algorithms (<code>LogisticRegression</code>, <code>RandomForestClassifier</code>), and performance metrics evaluation (<code>classification_report</code>, <code>confusion_matrix</code>, <code>roc_curve</code>, <code>f1_score</code>, <code>auc</code>).</li>
        <li><strong>Explainable AI (XAI):</strong> <code>shap</code> (SHapley Additive exPlanations) using its optimized <code>TreeExplainer</code> module to calculate global variable effects and local individual prediction waterfall graphs.</li>
    </ul>

    <h2>🔍 3. Data Exploration & Quality Check</h2>
    <p>
        Before writing any machine learning models, an exhaustive exploratory assessment was conducted on the source file <code>bank-full.csv</code>:
    </p>
    <ul>
        <li><strong>Shape & Volumetrics:</strong> The dataset contains <strong>45,211 rows</strong> and <strong>17 base columns</strong>.</li>
        <li><strong>Missing Values Check:</strong> Every single column was audited via <code>df.isnull().sum()</code>. The total count of missing or null fields was exactly <strong>0</strong>, meaning no imputation strategies were required.</li>
        <li><strong>Duplicated Records:</strong> Checked via <code>df.duplicated().sum()</code>. No duplicate records were found, proving the database rows are unique customer records.</li>
        <li><strong>Class Imbalance:</strong> The target variable <code>y</code> revealed an intense class skew:
            <table>
                <thead>
                    <tr>
                        <th>Target Outcome (y)</th>
                        <th>Absolute Count</th>
                        <th>Percentage Share</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>No (Refused Deposit)</strong></td>
                        <td>39,922</td>
                        <td>88.30%</td>
                    </tr>
                    <tr>
                        <td><strong>Yes (Subscribed Deposit)</strong></td>
                        <td>5,289</td>
                        <td>11.70%</td>
                    </tr>
                </tbody>
            </table>
            This imbalance requires metrics beyond simple accuracy, such as the <strong>F1-Score</strong> and <strong>AUC-ROC</strong>, to judge model validity fairly.
        </li>
    </ul>

    <h2>📊 4. Exploratory Data Analysis (EDA) — Key Insights</h2>
    <p>
        Seven analytical visualization blocks were executed to uncover deep behavioral patterns:
    </p>
    <ol>
        <li><strong>Target Skew:</strong> Formally verified the ~11.7% subscription baseline.</li>
        <li><strong>Age Group Fluctuations:</strong> Plotting distributions for subscribers vs non-subscribers revealed non-linear patterns. While the bulk of contacts are aged 30–50, the highest conversion ratios are clustered among <strong>young adults (under 25)</strong> and <strong>seniors (aged 60+)</strong>, likely reflecting distinct liquidity needs and free time to interact with bank agents.</li>
        <li><strong>Occupational Discrepancies:</strong> Categorizing subscriptions by <code>job</code> showed that <strong>retired individuals</strong> and <strong>students</strong> convert at the highest rates. Conversely, blue-collar workers and entrepreneurs showed low relative conversion.</li>
        <li><strong>Marital & Educational Trends:</strong> Single individuals convert at higher margins than married individuals. Furthermore, individuals holding <strong>tertiary education</strong> display a higher propensity to subscribe compared to those with primary or secondary schooling.</li>
        <li><strong>Campaign Call Overkill:</strong> Analyzing the number of contacts made during the campaign (<code>campaign</code>) revealed a point of diminishing returns. The probability of subscription peaks on the 1st or 2nd call and decays rapidly thereafter. Contacting a customer more than 4–5 times yields negligible returns and increases irritation.</li>
        <li><strong>The Power of Call Duration:</strong> The numerical feature <code>duration</code> (length of the phone call in seconds) shows an extraordinarily strong positive correlation with subscription. Long call lengths signify deep customer engagement, turning the telephone conversation into a critical diagnostic indicator.</li>
        <li><strong>Temporal/Monthly Variations:</strong> Grouping conversions by <code>month</code> revealed massive anomalies. While the bulk of absolute calls are pushed in <strong>May</strong>, the success rate in May is extremely poor. In contrast, off-peak months like <strong>March, September, October, and December</strong> show massive subscription ratios, potentially aligning with quarterly financial allocations and fiscal closures.</li>
    </ol>

    <h2>🔧 5. Robust Data Preprocessing Pipeline</h2>
    <p>
        To transition raw data into algorithmic inputs, a meticulous encoding and scaling strategy was implemented:
    </p>
    <ul>
        <li><strong>Target Mapping:</strong> The label vector <code>y</code> was mapped directly: <code>'yes' &rarr; 1</code> and <code>'no' &rarr; 0</code>.</li>
        <li><strong>Binary Variable Conversion:</strong> Natural binary inputs (<code>default</code>, <code>housing</code>, <code>loan</code>) containing text strings were transformed into <code>1</code> (Yes) and <code>0</code> (No).</li>
        <li><strong>Multi-Class Categorical Expansion:</strong> Categorical variables with multiple levels (<code>job</code>, <code>marital</code>, <code>education</code>, <code>contact</code>, <code>poutcome</code>, <code>month</code>) were processed using <strong>One-Hot Encoding</strong> via <code>pd.get_dummies</code>. To prevent the "dummy variable trap" (multicollinearity), <code>drop_first=True</code> was actively enforced. This expanded the feature set from 16 predictors to <strong>42 distinct numerical columns</strong>.</li>
        <li><strong>Feature Scaling:</strong> Since algorithms like Logistic Regression calculate coefficients based on numerical boundaries, all features were standardized using <code>StandardScaler</code>, shifting values to have a mean of 0 and a standard deviation of 1.</li>
    </ul>

    <h2>✂️ 6. Stratified Train-Test Splitting</h2>
    <p>
        To ensure unbiased testing, the 45,211 records were split into <strong>80% for model training</strong> and <strong>20% for final validation</strong>. Crucially, the split incorporated <code>stratify=y</code>, forcing both the training and test matrices to preserve the exact 11.70% subscription ratio, preventing class representation gaps across subsets.
    </p>
    <ul>
        <li><strong>Training Subset:</strong> 36,168 records (31,937 No / 4,231 Yes)</li>
        <li><strong>Testing Subset:</strong> 9,043 records (7,985 No / 1,058 Yes)</li>
    </ul>

    <h2>🤖 7. Model Selection & Strategy</h2>
    <p>Two contrasting machine learning philosophies were deployed to handle the task:</p>
    <div class="grid-2">
        <div style="background: var(--light-color); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color);">
            <h4 style="margin-top:0; color: var(--primary-color);">📈 Model A: Logistic Regression</h4>
            <p style="font-size: 0.9rem;">
                Acts as a linear baseline. It calculates log-odds coefficients for each feature. It is incredibly fast to train, completely transparent, and provides highly reliable probability thresholds. The optimization iterations parameter (<code>max_iter</code>) was adjusted to 1000 to guarantee absolute matrix convergence.
            </p>
        </div>
        <div style="background: var(--light-color); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color);">
            <h4 style="margin-top:0; color: var(--primary-color);">🌲 Model B: Random Forest Classifier</h4>
            <p style="font-size: 0.9rem;">
                An ensemble learning method consisting of 200 distinct decision trees. It constructs multiple non-linear boundary rules, aggregates votes across trees, and natively handles complex multi-column interactions without overfitting. Class weighting was utilized to address underlying data imbalances.
            </p>
        </div>
    </div>

    <h2>📈 8. Experimental Results & Performance Summary</h2>
    <p>
        After training, both models evaluated the 9,043 independent test samples. The comprehensive metrics highlight a clear winner:
    </p>
    
    <table>
        <thead>
            <tr>
                <th>Classification Model</th>
                <th>Overall Accuracy</th>
                <th>Weighted F1-Score</th>
                <th>Area Under ROC Curve (AUC-ROC)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Baseline Logistic Regression</strong></td>
                <td>84.60%</td>
                <td>0.8655</td>
                <td>0.9079</td>
            </tr>
            <tr style="background-color: #f0fff4; font-weight: bold; border-left: 4px solid var(--secondary-color);">
                <td>🏆 Random Forest Classifier</td>
                <td>87.89%</td>
                <td>0.8899</td>
                <td>0.9241</td>
            </tr>
        </tbody>
    </table>

    <div class="grid-2">
        <div>
            <h3>LOGISTIC REGRESSION Report</h3>
            <pre class="code-block">
              precision    recall  f1-score   support

      No (0)       0.97      0.85      0.91      7985
     Yes (1)       0.42      0.81      0.55      1058

    accuracy                           0.85      9043
            </pre>
        </div>
        <div>
            <h3>RANDOM FOREST Report</h3>
            <pre class="code-block">
              precision    recall  f1-score   support

      No (0)       0.97      0.89      0.93      7985
     Yes (1)       0.49      0.76      0.60      1058

    accuracy                           0.88      9043
            </pre>
        </div>
    </div>

    <div class="insight-box">
        <h4>Key Evaluative Takeaways:</h4>
        <ul>
            <li><strong>Random Forest outpaced Logistic Regression</strong> across all fundamental parameters, hitting an exceptional <strong>AUC-ROC of 0.9241</strong>, establishing an elite capacity to separate true subscribers from non-subscribers.</li>
            <li>Both models exhibit excellent precision (97%) for identifying non-subscribers, ensuring the bank rarely misclassifies a non-interested lead as interested.</li>
            <li>Random Forest successfully balances minority class precision (49%) and recall (76%), boosting the total F1-Score to <strong>0.8899</strong>.</li>
        </ul>
    </div>

    <h2>🔍 9. Explainable AI (XAI) with SHAP</h2>
    <p>
        Modern banks cannot rely purely on "black-box" models due to regulatory and strategic demands. To explain the model's inner reasoning, <strong>SHAP (SHapley Additive exPlanations)</strong> values were computed via <code>TreeExplainer</code>.
    </p>
    <ul>
        <li><strong>Global Feature Importance:</strong> SHAP calculations mathematically confirmed that <strong><code>duration</code> (call duration) is the number-one driver</strong> of the outcome. The longer an agent maintains active verbal contact with a client, the higher the mathematical attribution score pushes toward a positive subscription prediction.</li>
        <li><strong>Directional Impact Matrix:</strong> Summary dot plots showed that high values of <code>poutcome_success</code> (meaning the client successfully subscribed in a prior campaign) have a massive rightward push, making it an incredibly strong secondary indicator of future conversion.</li>
        <li><strong>Negative Pushes:</strong> High values of <code>housing</code> (having an active housing loan) or <code>loan_yes</code> (having a personal loan) generate negative SHAP values, showing that heavy existing debt significantly reduces a customer's willingness or financial capacity to open a new term deposit.</li>
    </ul>

    <h2>💡 10. Core Strategic Business Recommendations</h2>
    <div class="highlight-box" style="background-color: #fffaf0; border-left-color: #dd6b20;">
        <h4>🎯 1. Smarter Lead Segmentation</h4>
        <p style="margin: 0; font-size: 0.95rem;">
            Stop calling the general population. Filter the database to highlight <strong>retired individuals, students, single clients, and individuals with tertiary degrees</strong>. Focus the primary campaign budget on these key demographics to capture easy conversions.
        </p>
    </div>
    
    <div class="highlight-box" style="background-color: #fffaf0; border-left-color: #dd6b20;">
        <h4>📞 2. Optimizing Contact Frequency & Quality</h4>
        <p style="margin: 0; font-size: 0.95rem;">
            Enforce a strict cap on call frequency. If a customer does not subscribe after <strong>



Gemini is AI and can make mistakes.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Term Deposit Subscription Prediction - README</title>
    <style>
        :root {
            --primary-color: #2b4c7e;
            --secondary-color: #2ecc71;
            --dark-color: #2c3e50;
            --light-color: #f8f9fa;
            --border-color: #e2e8f0;
            --text-muted: #718096;
            --accent-color: #e74c3c;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            background-color: #f4f6f9;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1100px;
            margin: 40px auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }

        .header {
            text-align: center;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 30px;
            margin-bottom: 40px;
        }

        .header h1 {
            color: var(--primary-color);
            margin: 0 0 10px 0;
            font-size: 2.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        .header h3 {
            color: var(--text-muted);
            font-weight: 400;
            margin: 0;
            font-size: 1.2rem;
        }

        .project-meta {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
            background: var(--light-color);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .meta-item {
            text-align: center;
        }

        .meta-item strong {
            display: block;
            color: var(--primary-color);
            font-size: 0.9rem;
            text-transform: uppercase;
            margin-bottom: 5px;
        }

        .meta-item span {
            font-size: 1.1rem;
            font-weight: 600;
        }

        h2 {
            color: var(--primary-color);
            border-left: 5px solid var(--primary-color);
            padding-left: 15px;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.75rem;
        }

        h3 {
            color: var(--dark-color);
            margin-top: 25px;
            font-size: 1.3rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 5px;
        }

        p {
            margin-bottom: 20px;
            text-align: justify;
        }

        ul, ol {
            margin-bottom: 20px;
            padding-left: 25px;
        }

        li {
            margin-bottom: 8px;
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #fff;
        }

        .badge-success { background-color: var(--secondary-color); }
        .badge-danger { background-color: var(--accent-color); }
        .badge-primary { background-color: var(--primary-color); }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95rem;
        }

        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
        }

        tr:hover {
            background-color: rgba(43, 76, 126, 0.02);
        }

        .highlight-box {
            background-color: #ebf4ff;
            border-left: 4px solid #3182ce;
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin: 25px 0;
        }

        .insight-box {
            background-color: #f0fff4;
            border-left: 4px solid var(--secondary-color);
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin: 25px 0;
        }

        .code-block {
            background-color: #1a202c;
            color: #edf2f7;
            padding: 15px 20px;
            border-radius: 6px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            margin: 20px 0;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 25px;
        }

        footer {
            text-align: center;
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>🏦 Term Deposit Subscription Prediction</h1>
        <h3>Bank Marketing Campaigns — Data Science Internship Project</h3>
        
        <div class="project-meta">
            <div class="meta-item">
                <strong>Dataset Source</strong>
                <span>UCI Repository</span>
            </div>
            <div class="meta-item">
                <strong>Records Count</strong>
                <span>45,211 rows</span>
            </div>
            <div class="meta-item">
                <strong>Best Performing Model</strong>
                <span>Random Forest</span>
            </div>
            <div class="meta-item">
                <strong>Top Predictor</strong>
                <span>Call Duration (`duration`)</span>
            </div>
            <div class="meta-item">
                <strong>Explainability Method</strong>
                <span>SHAP (TreeExplainer)</span>
            </div>
        </div>
    </div>

    <h2>📌 1. Project Overview & Problem Statement</h2>
    <p>
        Retail banking organizations frequently conduct direct telemarketing campaigns to encourage clients to subscribe to long-term financial products, specifically <strong>term deposits</strong>. A term deposit is a fixed-term investment where a customer locks away an amount of capital for a specified duration at a guaranteed interest rate. For banks, securing these deposits is vital because it provides stable, low-cost capital to fund lending activities and financial products.
    </p>
    <p>
        However, traditional phone-based marketing strategies suffer from immense inefficiency. Cold-calling or blindly contacting every customer inside a large database results in massive resource wastage (high telecom fees, agent fatigue, operational overhead) and causes customer annoyance, potentially damaging customer relations. 
    </p>
    <div class="highlight-box">
        <strong>The Core Challenge:</strong> Out of 45,211 customers contacted in this historic campaign dataset, only <strong>5,289 (11.70%)</strong> actually subscribed to the term deposit. The remaining 88.30% represented wasted effort.
    </div>
    <p>
        <strong>Objective:</strong> The primary objective of this project is to develop a robust binary classification pipeline that leverages machine learning to predict whether a customer will subscribe (<code class="badge badge-primary">yes</code> / 1) or refuse (<code class="badge badge-danger">no</code> / 0) a term deposit. By filtering leads and scoring customers beforehand, the marketing team can prioritize high-probability converts, maximize conversion rates, and minimize operational costs.
    </p>

    <h2>📦 2. Technical Stack & Libraries Used</h2>
    <p>
        The complete pipeline was developed using Python's extensive data science ecosystem. The primary components include:
    </p>
    <ul>
        <li><strong>Data Manipulation & Numerical Operations:</strong> <code>pandas</code> and <code>numpy</code> for reading data, slicing, dtypes inspection, handling tabular frames, and matrix formatting.</li>
        <li><strong>Data Visualization:</strong> <code>matplotlib.pyplot</code> and <code>seaborn</code> to build professional distributions, side-by-side counts, correlation matrices, and customized metric graphs.</li>
        <li><strong>Machine Learning Pipeline:</strong> <code>scikit-learn</code> for data preprocessing (<code>StandardScaler</code>), dataset splitting (<code>train_test_split</code>), classification algorithms (<code>LogisticRegression</code>, <code>RandomForestClassifier</code>), and performance metrics evaluation (<code>classification_report</code>, <code>confusion_matrix</code>, <code>roc_curve</code>, <code>f1_score</code>, <code>auc</code>).</li>
        <li><strong>Explainable AI (XAI):</strong> <code>shap</code> (SHapley Additive exPlanations) using its optimized <code>TreeExplainer</code> module to calculate global variable effects and local individual prediction waterfall graphs.</li>
    </ul>

    <h2>🔍 3. Data Exploration & Quality Check</h2>
    <p>
        Before writing any machine learning models, an exhaustive exploratory assessment was conducted on the source file <code>bank-full.csv</code>:
    </p>
    <ul>
        <li><strong>Shape & Volumetrics:</strong> The dataset contains <strong>45,211 rows</strong> and <strong>17 base columns</strong>.</li>
        <li><strong>Missing Values Check:</strong> Every single column was audited via <code>df.isnull().sum()</code>. The total count of missing or null fields was exactly <strong>0</strong>, meaning no imputation strategies were required.</li>
        <li><strong>Duplicated Records:</strong> Checked via <code>df.duplicated().sum()</code>. No duplicate records were found, proving the database rows are unique customer records.</li>
        <li><strong>Class Imbalance:</strong> The target variable <code>y</code> revealed an intense class skew:
            <table>
                <thead>
                    <tr>
                        <th>Target Outcome (y)</th>
                        <th>Absolute Count</th>
                        <th>Percentage Share</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>No (Refused Deposit)</strong></td>
                        <td>39,922</td>
                        <td>88.30%</td>
                    </tr>
                    <tr>
                        <td><strong>Yes (Subscribed Deposit)</strong></td>
                        <td>5,289</td>
                        <td>11.70%</td>
                    </tr>
                </tbody>
            </table>
            This imbalance requires metrics beyond simple accuracy, such as the <strong>F1-Score</strong> and <strong>AUC-ROC</strong>, to judge model validity fairly.
        </li>
    </ul>

    <h2>📊 4. Exploratory Data Analysis (EDA) — Key Insights</h2>
    <p>
        Seven analytical visualization blocks were executed to uncover deep behavioral patterns:
    </p>
    <ol>
        <li><strong>Target Skew:</strong> Formally verified the ~11.7% subscription baseline.</li>
        <li><strong>Age Group Fluctuations:</strong> Plotting distributions for subscribers vs non-subscribers revealed non-linear patterns. While the bulk of contacts are aged 30–50, the highest conversion ratios are clustered among <strong>young adults (under 25)</strong> and <strong>seniors (aged 60+)</strong>, likely reflecting distinct liquidity needs and free time to interact with bank agents.</li>
        <li><strong>Occupational Discrepancies:</strong> Categorizing subscriptions by <code>job</code> showed that <strong>retired individuals</strong> and <strong>students</strong> convert at the highest rates. Conversely, blue-collar workers and entrepreneurs showed low relative conversion.</li>
        <li><strong>Marital & Educational Trends:</strong> Single individuals convert at higher margins than married individuals. Furthermore, individuals holding <strong>tertiary education</strong> display a higher propensity to subscribe compared to those with primary or secondary schooling.</li>
        <li><strong>Campaign Call Overkill:</strong> Analyzing the number of contacts made during the campaign (<code>campaign</code>) revealed a point of diminishing returns. The probability of subscription peaks on the 1st or 2nd call and decays rapidly thereafter. Contacting a customer more than 4–5 times yields negligible returns and increases irritation.</li>
        <li><strong>The Power of Call Duration:</strong> The numerical feature <code>duration</code> (length of the phone call in seconds) shows an extraordinarily strong positive correlation with subscription. Long call lengths signify deep customer engagement, turning the telephone conversation into a critical diagnostic indicator.</li>
        <li><strong>Temporal/Monthly Variations:</strong> Grouping conversions by <code>month</code> revealed massive anomalies. While the bulk of absolute calls are pushed in <strong>May</strong>, the success rate in May is extremely poor. In contrast, off-peak months like <strong>March, September, October, and December</strong> show massive subscription ratios, potentially aligning with quarterly financial allocations and fiscal closures.</li>
    </ol>

    <h2>🔧 5. Robust Data Preprocessing Pipeline</h2>
    <p>
        To transition raw data into algorithmic inputs, a meticulous encoding and scaling strategy was implemented:
    </p>
    <ul>
        <li><strong>Target Mapping:</strong> The label vector <code>y</code> was mapped directly: <code>'yes' &rarr; 1</code> and <code>'no' &rarr; 0</code>.</li>
        <li><strong>Binary Variable Conversion:</strong> Natural binary inputs (<code>default</code>, <code>housing</code>, <code>loan</code>) containing text strings were transformed into <code>1</code> (Yes) and <code>0</code> (No).</li>
        <li><strong>Multi-Class Categorical Expansion:</strong> Categorical variables with multiple levels (<code>job</code>, <code>marital</code>, <code>education</code>, <code>contact</code>, <code>poutcome</code>, <code>month</code>) were processed using <strong>One-Hot Encoding</strong> via <code>pd.get_dummies</code>. To prevent the "dummy variable trap" (multicollinearity), <code>drop_first=True</code> was actively enforced. This expanded the feature set from 16 predictors to <strong>42 distinct numerical columns</strong>.</li>
        <li><strong>Feature Scaling:</strong> Since algorithms like Logistic Regression calculate coefficients based on numerical boundaries, all features were standardized using <code>StandardScaler</code>, shifting values to have a mean of 0 and a standard deviation of 1.</li>
    </ul>

    <h2>✂️ 6. Stratified Train-Test Splitting</h2>
    <p>
        To ensure unbiased testing, the 45,211 records were split into <strong>80% for model training</strong> and <strong>20% for final validation</strong>. Crucially, the split incorporated <code>stratify=y</code>, forcing both the training and test matrices to preserve the exact 11.70% subscription ratio, preventing class representation gaps across subsets.
    </p>
    <ul>
        <li><strong>Training Subset:</strong> 36,168 records (31,937 No / 4,231 Yes)</li>
        <li><strong>Testing Subset:</strong> 9,043 records (7,985 No / 1,058 Yes)</li>
    </ul>

    <h2>🤖 7. Model Selection & Strategy</h2>
    <p>Two contrasting machine learning philosophies were deployed to handle the task:</p>
    <div class="grid-2">
        <div style="background: var(--light-color); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color);">
            <h4 style="margin-top:0; color: var(--primary-color);">📈 Model A: Logistic Regression</h4>
            <p style="font-size: 0.9rem;">
                Acts as a linear baseline. It calculates log-odds coefficients for each feature. It is incredibly fast to train, completely transparent, and provides highly reliable probability thresholds. The optimization iterations parameter (<code>max_iter</code>) was adjusted to 1000 to guarantee absolute matrix convergence.
            </p>
        </div>
        <div style="background: var(--light-color); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color);">
            <h4 style="margin-top:0; color: var(--primary-color);">🌲 Model B: Random Forest Classifier</h4>
            <p style="font-size: 0.9rem;">
                An ensemble learning method consisting of 200 distinct decision trees. It constructs multiple non-linear boundary rules, aggregates votes across trees, and natively handles complex multi-column interactions without overfitting. Class weighting was utilized to address underlying data imbalances.
            </p>
        </div>
    </div>

    <h2>📈 8. Experimental Results & Performance Summary</h2>
    <p>
        After training, both models evaluated the 9,043 independent test samples. The comprehensive metrics highlight a clear winner:
    </p>
    
    <table>
        <thead>
            <tr>
                <th>Classification Model</th>
                <th>Overall Accuracy</th>
                <th>Weighted F1-Score</th>
                <th>Area Under ROC Curve (AUC-ROC)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Baseline Logistic Regression</strong></td>
                <td>84.60%</td>
                <td>0.8655</td>
                <td>0.9079</td>
            </tr>
            <tr style="background-color: #f0fff4; font-weight: bold; border-left: 4px solid var(--secondary-color);">
                <td>🏆 Random Forest Classifier</td>
                <td>87.89%</td>
                <td>0.8899</td>
                <td>0.9241</td>
            </tr>
        </tbody>
    </table>

    <div class="grid-2">
        <div>
            <h3>LOGISTIC REGRESSION Report</h3>
            <pre class="code-block">
              precision    recall  f1-score   support

      No (0)       0.97      0.85      0.91      7985
     Yes (1)       0.42      0.81      0.55      1058

    accuracy                           0.85      9043
            </pre>
        </div>
        <div>
            <h3>RANDOM FOREST Report</h3>
            <pre class="code-block">
              precision    recall  f1-score   support

      No (0)       0.97      0.89      0.93      7985
     Yes (1)       0.49      0.76      0.60      1058

    accuracy                           0.88      9043
            </pre>
        </div>
    </div>

    <div class="insight-box">
        <h4>Key Evaluative Takeaways:</h4>
        <ul>
            <li><strong>Random Forest outpaced Logistic Regression</strong> across all fundamental parameters, hitting an exceptional <strong>AUC-ROC of 0.9241</strong>, establishing an elite capacity to separate true subscribers from non-subscribers.</li>
            <li>Both models exhibit excellent precision (97%) for identifying non-subscribers, ensuring the bank rarely misclassifies a non-interested lead as interested.</li>
            <li>Random Forest successfully balances minority class precision (49%) and recall (76%), boosting the total F1-Score to <strong>0.8899</strong>.</li>
        </ul>
    </div>

    <h2>🔍 9. Explainable AI (XAI) with SHAP</h2>
    <p>
        Modern banks cannot rely purely on "black-box" models due to regulatory and strategic demands. To explain the model's inner reasoning, <strong>SHAP (SHapley Additive exPlanations)</strong> values were computed via <code>TreeExplainer</code>.
    </p>
    <ul>
        <li><strong>Global Feature Importance:</strong> SHAP calculations mathematically confirmed that <strong><code>duration</code> (call duration) is the number-one driver</strong> of the outcome. The longer an agent maintains active verbal contact with a client, the higher the mathematical attribution score pushes toward a positive subscription prediction.</li>
        <li><strong>Directional Impact Matrix:</strong> Summary dot plots showed that high values of <code>poutcome_success</code> (meaning the client successfully subscribed in a prior campaign) have a massive rightward push, making it an incredibly strong secondary indicator of future conversion.</li>
        <li><strong>Negative Pushes:</strong> High values of <code>housing</code> (having an active housing loan) or <code>loan_yes</code> (having a personal loan) generate negative SHAP values, showing that heavy existing debt significantly reduces a customer's willingness or financial capacity to open a new term deposit.</li>
    </ul>

    <h2>💡 10. Core Strategic Business Recommendations</h2>
    <div class="highlight-box" style="background-color: #fffaf0; border-left-color: #dd6b20;">
        <h4>🎯 1. Smarter Lead Segmentation</h4>
        <p style="margin: 0; font-size: 0.95rem;">
            Stop calling the general population. Filter the database to highlight <strong>retired individuals, students, single clients, and individuals with tertiary degrees</strong>. Focus the primary campaign budget on these key demographics to capture easy conversions.
        </p>
    </div>
    
    <div class="highlight-box" style="background-color: #fffaf0; border-left-color: #dd6b20;">
        <h4>📞 2. Optimizing Contact Frequency & Quality</h4>
        <p style="margin: 0; font-size: 0.95rem;">
            Enforce a strict cap on call frequency. If a customer does not subscribe after <strong>2 to 3 contact attempts</strong>, flag them as "inactive" for the current cycle. Wasting a 4th or 5th call destroys staff time and decreases customer satisfaction. Instead, train agents on script quality to maximize <strong>call duration</strong> on initial touches.
        </p>
    </div>

    <div class="highlight-box" style="background-color: #fffaf0; border-left-color: #dd6b20;">
        <h4>🗓️ 3. Temporal Campaign Scheduling</h4>
        <p style="margin: 0; font-size: 0.95rem;">
            Re-allocate staff and budget from May to high-performing months like <strong>March, September, October, and December</strong>. These months align perfectly with end-of-quarter or year-end financial management cycles when clients actively seek safe yield opportunities like term deposits.
        </p>
    </div>

    <div class="highlight-box" style="background-color: #fffaf0; border-left-color: #dd6b20;">
        <h4>💳 4. Debt and Historical Filtering</h4>
        <p style="margin: 0; font-size: 0.95rem;">
            Exclude or de-prioritize individuals burdened with housing or personal loans. Concurrently, create an automated "Fast Track" list for any customer flagged as <code>poutcome_success</code> from prior outreach, as they represent highly receptive loyal leads.
        </p>
    </div>

    <h2>🏁 11. Final Project Conclusion & Next Steps</h2>
    <p>
        This end-to-end data science project proves that predictive modeling can turn a chaotic, low-yield marketing approach into an efficient, data-driven system. Moving from an 11.7% blind success rate to a structured machine learning pipeline backed by a <strong>0.9241 AUC-ROC Random Forest Model</strong> enables the bank to contact only the top 20–30% of high-probability clients. This optimization protects operating capital, boosts conversion margins, and protects brand reputation.
    </p>
    <p><strong>Proposed Future Enhancements:</strong></p>
    <ol>
        <li>Implement advanced boosting frameworks like <strong>XGBoost, LightGBM, or CatBoost</strong> to try and push the F1-score past 0.90.</li>
        <li>Utilize <strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> to balance the class representation during training.</li>
        <li>Package the trained Random Forest serialization object (<code>.pkl</code>) into a microservice API using <strong>FastAPI</strong>, allowing CRM systems to score prospects instantly during live operations.</li>
    </ol>

    <footer>
        <p>Bank Marketing Predictive Analytics Pipeline — Portfolio Project © 2026</p>
    </footer>
</div>

</body>
</html>
README.html
Displaying README.html.
