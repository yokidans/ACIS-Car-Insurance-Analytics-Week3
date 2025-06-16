# ACIS: Car Insurance Analytics 

## Project Overview
**AlphaCare Insurance Solutions (ACIS)** is undertaking a data-driven initiative to optimize car insurance marketing strategies and identify low-risk client segments in South Africa. This project leverages advanced analytics to transform raw insurance data into actionable business insights.

## Key Objectives
- 🎯 **Risk Profiling**: Quantify risk across demographic, geographic, and vehicle factors
- 💰 **Profitability Analysis**: Identify high-margin customer segments
- 📊 **Marketing Optimization**: Enable targeted campaigns using data-driven segmentation
- 🔮 **Premium Prediction**: Develop ML models to recommend optimal premium pricing

## Tasks
1. **Exploratory Data Analysis (EDA)**
   - Data quality assessment and cleaning
   - Univariate and multivariate analysis
   - Outlier detection (e.g., 2,550 policies with extreme loss ratios >2)
   - Key visualizations:
     - Log-transformed premium distributions
     - Premium analysis by vehicle type (top 10)
     - Geographic risk patterns

2. **A/B Hypothesis Testing**
   - Statistical validation of risk differences across:
     - Provinces (ANOVA/Kruskal-Wallis)
     - Zip codes
     - Gender groups (t-tests)
   - Margin analysis between segments

3. **Machine Learning Modeling**
   - Linear regression by zip code
   - Premium prediction using:
     - Random Forest (MAE: 2.02M, R²: 1.0)
     - XGBoost (MAE: 133.9M, R²: 0.832)
   - Feature importance analysis via SHAP values

4. **Reporting & Insights**
   - Executive summaries with actionable recommendations
   - Data visualization dashboards
   - Regulatory compliance documentation
