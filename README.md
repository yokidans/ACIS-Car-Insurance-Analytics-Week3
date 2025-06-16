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

## Technical Implementation
### Data Pipeline
# DVC Workflow Example
    dvc init
    dvc remote add -d localstorage /path/to/storage
    dvc add data/raw/insurance_claims.csv
    git add .gitignore data/raw/insurance_claims.csv.dvc
    dvc push

 ## Key Findings
### Risk Hotspots:

- 83.1% of high-loss policies involve passenger vehicles

- Specific provinces show 48.2% concentration of high-risk policies

### Premium Patterns:

- Right-skewed distribution (log-transform required)

- Strong correlation between sum insured and premiums

### Demographic Insights:

- Age groups 25-35 show highest risk (41.3% of high-loss policies)

  ### ACIS-Car-Insurance-Analytics/
### ├── data/
### │   ├── raw/               # Original datasets
### │   └── processed/         # Cleaned data (1.5M → 151K records after deduplication)
### ├── docs/                  # Project documentation
### ├── notebooks/             # Jupyter notebooks for EDA
### ├── scripts/
### │   ├── data_processing.py # Handles missing values (e.g., 779K missing custom values)
### │   └── modeling.py        # ML pipeline (Linear Regression → XGBoost)
### └── params.yaml            # Model hyperparameters

# How to Reproduce
## 1. Install dependencies:
    pip install -r requirements.txt
    pip install dvc
## 2. Run data pipeline:
    dvc repro
## 3. Execute analysis notebooks

## Key Business Impact
- 15-20% potential reduction in high-risk policy acquisitions

- 5-8% improvement in premium pricing accuracy

- Identified 3 vehicle types for targeted marketing campaigns

Last Updated: June 2025 | Data Version: v3.1.2 | [ACIS Regulatory Compliance]
