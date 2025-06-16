# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import json
import os
from sklearn.impute import SimpleImputer

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_absolute_error, 
                           mean_squared_error, 
                           r2_score,
                           make_scorer)

# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap

# Configuration
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.random.seed(42)

# Load parameters from params.yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)

fe_params = params['feature_engineering']
model_params = params['model']

# Load processed data
df = pd.read_csv(fe_params['input_file'])

# Feature Selection
features = fe_params['features']
target = fe_params['target']

# Filter available features
available_features = [f for f in features if f in df.columns]
X = df[available_features]
y = df[target]

# Train-test split (stratified by claim amount bins)
bins = pd.qcut(y, q=5, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=fe_params['test_size'], 
    stratify=bins, 
    random_state=fe_params['random_state']
)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")

# Numeric vs Categorical features
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Pipeline construction
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Handle numeric NaNs
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle categorical NaNs
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])
# Check data quality
print("\nData Quality Report:")
print("Missing Values:")
print(X_train.isna().sum())
print("\nData Types:")
print(X_train.dtypes)

# Model Definitions
models = {
    "Linear Regression": Pipeline([
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer()),  # Additional safety
        ('model', LinearRegression())
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=fe_params.get('random_state', 42),
        n_jobs=-1  # Enable parallel processing
    ),
    "XGBoost": XGBRegressor(
        objective=model_params.get('objective', 'reg:squarederror'),
        n_estimators=model_params.get('n_estimators', 100),
        max_depth=model_params.get('max_depth', 6),
        learning_rate=model_params.get('learning_rate', 0.1),
        random_state=fe_params.get('random_state', 42),
        enable_categorical=True  # Handle categoricals directly
    )
}

# Evaluation Function
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Check for NaN values
    print(f"NaN values in X_train: {X_train.isna().sum().sum()}")
    print(f"NaN values in y_train: {y_train.isna().sum()}")
    
    if isinstance(model, Pipeline):
        pipeline = model
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return pipeline, metrics

# Model Comparison
results = {}
for name, model in tqdm(models.items()):
    pipeline, metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = {
        'model': pipeline,
        'metrics': metrics
    }
    print(f"\n{name} Performance:")
    print(f"- MAE: {metrics['MAE']:,.2f}")
    print(f"- RMSE: {metrics['RMSE']:,.2f}")
    print(f"- R2: {metrics['R2']:.3f}")

# Save metrics
os.makedirs('../reports', exist_ok=True)
with open('../reports/evaluation_metrics.json', 'w') as f:
    json.dump(results['XGBoost']['metrics'], f)

# Feature Importance Analysis
best_model = results['XGBoost']['model']
X_test['predicted_claims'] = best_model.predict(X_test)
X_test['actual_claims'] = y_test.values

# Save predictions
os.makedirs('../data/outputs', exist_ok=True)
X_test[['predicted_claims', 'actual_claims']].to_csv(fe_params['output_file'])

print("Pipeline execution completed successfully!")