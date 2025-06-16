# =============================================
# AlphaCare Insurance Solutions (ACIS) - Predictive Analytics
# =============================================
# Step-by-Step Implementation:
# 1. Data Loading & Initial Inspection
# 2. Data Cleaning & Missing Value Handling
# 3. Exploratory Data Analysis (EDA)
# 4. Hypothesis Testing (A/B Tests)
# 5. Feature Engineering
# 6. Predictive Modeling
# 7. Results Interpretation
# =============================================

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_verify_data(filepath):
    """Complete data loading solution with verification"""
    print("\n=== DATA LOADING PROCESS ===")
    
    # Step 1: Load raw data
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load CSV: {str(e)}")
        return None
    
    # Step 2: Verify critical columns
    required_columns = {
        'premium': ['TotalPremium', 'Premium', 'Amount'],
        'claims': ['TotalClaims', 'Claims', 'ClaimAmount'],
        'date': ['TransactionMonth', 'Date', 'Month', 'TransactionDate']
    }
    
    # Map actual columns to expected names
    column_map = {}
    for expected_type, possible_names in required_columns.items():
        for name in possible_names:
            if name in df.columns:
                column_map[expected_type] = name
                break
    
    # Step 3: Handle missing columns
    if not column_map.get('premium'):
        print("\n❌ Error: No premium amount column found")
        print("Available columns:", df.columns.tolist())
        return None
    
    if not column_map.get('claims'):
        print("\n❌ Error: No claims amount column found")
        return None
    
    # Step 4: Process dates if available
    if column_map.get('date'):
        date_col = column_map['date']
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        print(f"✅ Converted '{date_col}' to datetime format")
    else:
        print("⚠️ Warning: No date column identified - temporal analysis disabled")
    
    return df

# Execute the loading process
df = load_and_verify_data('insurance_claims.csv')

if df is not None:
    print("\n=== DATA READY FOR ANALYSIS ===")
    print(df.info())
else:
    print("\n❌ Cannot proceed - please fix your input file")
# Continue with rest of your analysis...
# ======================
# 3. EXPLORATORY DATA ANALYSIS
# ======================
print("\n=== STEP 3: EXPLORATORY DATA ANALYSIS ===")

# 3.1 Univariate Analysis
print("\n--- Univariate Analysis ---")

# Numerical distributions
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['TotalPremium'], kde=True, color='blue')
plt.title('Total Premium Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['TotalClaims'], kde=True, color='red')
plt.title('Total Claims Distribution')
plt.tight_layout()
plt.show()

# Categorical distributions
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
df['Gender'].value_counts().plot(kind='bar', color='green')
plt.title('Gender Distribution')

plt.subplot(1, 2, 2)
df['Province'].value_counts().plot(kind='bar', color='purple')
plt.title('Province Distribution')
plt.tight_layout()
plt.show()

# 3.2 Bivariate Analysis
print("\n--- Bivariate Analysis ---")

# Calculate Loss Ratio
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']

# Loss Ratio by Province
plt.figure(figsize=(14, 6))
sns.barplot(x='Province', y='LossRatio', data=df, palette='viridis')
plt.title('Loss Ratio by Province (Higher = More Risky)')
plt.xticks(rotation=45)
plt.show()

# Correlation Analysis
plt.figure(figsize=(10, 8))
corr_matrix = df[['TotalPremium', 'TotalClaims', 'CustomValueEstimate']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# 3.3 Temporal Analysis
print("\n--- Temporal Analysis ---")

# Monthly trends
monthly_data = df.groupby('TransactionMonth').agg({
    'TotalPremium': 'sum',
    'TotalClaims': 'sum'
}).reset_index()

plt.figure(figsize=(14, 6))
plt.plot(monthly_data['TransactionMonth'], monthly_data['TotalPremium'], label='Total Premium')
plt.plot(monthly_data['TransactionMonth'], monthly_data['TotalClaims'], label='Total Claims')
plt.title('Monthly Premium vs Claims Trend')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()

# ======================
# 4. HYPOTHESIS TESTING
# ======================
print("\n=== STEP 4: HYPOTHESIS TESTING ===")

# Hypothesis 1: Risk differences across provinces
print("\n--- Hypothesis 1: Province Risk Differences ---")
provinces = df['Province'].unique()
province_groups = [df[df['Province'] == prov]['LossRatio'] for prov in provinces]
f_stat, p_value = stats.f_oneway(*province_groups)
print(f"ANOVA p-value: {p_value:.4f}")
print("Conclusion:", "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis")

# Hypothesis 2: Gender risk differences
print("\n--- Hypothesis 2: Gender Risk Differences ---")
male_loss = df[df['Gender'] == 'Male']['LossRatio']
female_loss = df[df['Gender'] == 'Female']['LossRatio']
t_stat, p_value = stats.ttest_ind(male_loss, female_loss)
print(f"T-test p-value: {p_value:.4f}")
print("Conclusion:", "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis")

# ======================
# 5. FEATURE ENGINEERING
# ======================
print("\n=== STEP 5: FEATURE ENGINEERING ===")

# Create new features
df['VehicleAge'] = df['TransactionMonth'].dt.year - df['RegistrationYear']
df['ProfitMargin'] = df['TotalPremium'] - df['TotalClaims']

# Encode categorical variables
label_encoders = {}
for col in ['Make', 'Model', 'Gender', 'Province']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Select features and target
features = ['Make', 'Model', 'Gender', 'Province', 'VehicleAge', 'SumInsured']
X = df[features]
y = df['TotalClaims']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# 6. PREDICTIVE MODELING
# ======================
print("\n=== STEP 6: PREDICTIVE MODELING ===")

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R²': r2, 'RMSE': rmse}

# Display results
results_df = pd.DataFrame(results).T
print("\n=== Model Performance ===")
print(results_df)

# Feature Importance from Random Forest
rf = models["Random Forest"]
importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title('Random Forest Feature Importance')
plt.show()

# ======================
# 7. INSIGHTS & RECOMMENDATIONS
# ======================
print("\n=== STEP 7: KEY INSIGHTS ===")

# 1. Risk Analysis
high_risk_provinces = df.groupby('Province')['LossRatio'].mean().nlargest(3)
print("\n🔴 Top 3 High-Risk Provinces:")
print(high_risk_provinces)

# 2. Vehicle Risk Analysis
vehicle_risk = df.groupby('Make')['TotalClaims'].mean().nlargest(5)
print("\n🚗 Top 5 High-Claim Vehicle Makes:")
print(vehicle_risk)

# 3. Profitability Analysis
profitable_zips = df.groupby('PostalCode')['ProfitMargin'].mean().nlargest(5)
print("\n💰 Top 5 Most Profitable Zip Codes:")
print(profitable_zips)

# Final Recommendations
print("\n✅ RECOMMENDATIONS:")
print("1. Increase premiums in high-risk provinces:", list(high_risk_provinces.index))
print("2. Offer discounts for low-risk vehicle makes")
print("3. Target marketing campaigns in profitable zip codes")
print("4. Investigate why certain vehicle makes have higher claims")

print("\n🎉 Analysis Complete!")