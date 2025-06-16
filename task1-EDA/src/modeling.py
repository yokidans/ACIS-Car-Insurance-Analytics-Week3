# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import json
import os
import os
from pathlib import Path
from sklearn.impute import SimpleImputer

# Define the base directory (assuming script runs from task1-EDA/src/)
BASE_DIR = Path(__file__).parent.parent.parent
# Enable iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Configuration
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.random.seed(42)

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                         np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
X = df[available_features].copy()  # Explicit copy to avoid SettingWithCopyWarning
y = df[target]

# Check data quality
print("\nData Quality Report:")
print("Missing Values:")
print(X.isna().sum())
print("\nData Types:")
print(X.dtypes)

# Handle missing values
print("\nFixing missing values...")
print(f"Initial missing values: {X.isna().sum().sum()}")

if X.isna().sum().sum() > 0:
    imputer = IterativeImputer(random_state=42)
    numeric_cols = X.select_dtypes(include=['number']).columns
    X.loc[:, numeric_cols] = imputer.fit_transform(X[numeric_cols])
    
print(f"Remaining missing values: {X.isna().sum().sum()}")

# Train-test split (stratified by claim amount bins)
bins = pd.qcut(y, q=5, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=fe_params['test_size'], 
    stratify=bins, 
    random_state=fe_params['random_state']
)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")

# Numeric vs Categorical features
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Pipeline construction
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Ensure we maintain DataFrames with proper column names
    X_train_df = pd.DataFrame(X_train, columns=features)
    X_test_df = pd.DataFrame(X_test, columns=features)
    
    if isinstance(model, Pipeline):
        pipeline = model
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    
    pipeline.fit(X_train_df, y_train)
    y_pred = pipeline.predict(X_test_df)
    
    # Convert numpy types to native Python types for JSON serialization
    metrics = {
        'MAE': float(mean_absolute_error(y_test, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'R2': float(r2_score(y_test, y_pred)),
        'Mean_Actual': float(np.mean(y_test)),
        'Mean_Predicted': float(np.mean(y_pred))
    }
    
    print("\nPrediction Analysis:")
    print(f"Actual mean: {metrics['Mean_Actual']:,.2f}")
    print(f"Predicted mean: {metrics['Mean_Predicted']:,.2f}")
    
    return pipeline, metrics, y_pred

# Model Definitions
models = {
    "Linear_Regression": Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ]),
    "Random_Forest": Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=fe_params['random_state'],
            n_jobs=-1
        ))
    ]),
    "XGBoost": Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(
            objective=model_params.get('objective', 'reg:squarederror'),
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.1),
            random_state=fe_params['random_state']
        ))
    ])
}

# Model Comparison
results = {}
for name, model in tqdm(models.items(), desc="Training Models"):
    try:
        pipeline, metrics, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = {
            'metrics': metrics,
            'predictions': y_pred.tolist()  # Convert numpy array to list
        }
        print(f"\n{name} Performance:")
        print(f"- MAE: {metrics['MAE']:,.2f}")
        print(f"- RMSE: {metrics['RMSE']:,.2f}")
        print(f"- R2: {metrics['R2']:.3f}")
    except Exception as e:
        print(f"\nFailed to train {name}: {str(e)}")
        continue

# Save outputs
os.makedirs('../reports', exist_ok=True)
os.makedirs('../data/outputs', exist_ok=True)

# Save metrics with custom encoder
with open('../reports/evaluation_metrics.json', 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

# Save predictions
predictions_df = pd.DataFrame({
    'actual_claims': y_test,
    'predicted_claims': results['XGBoost']['predictions']
})
# predictions_df.to_csv(fe_params['output_file'], index=False)
metrics_path = BASE_DIR / 'reports' / 'evaluation_metrics.json'
metrics_path.parent.mkdir(parents=True, exist_ok=True)
with open(metrics_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)
print(f"Saved metrics to: {metrics_path}")
# Save predictions
predictions_path = BASE_DIR / 'data' / 'outputs' / 'predictions.csv'
predictions_path.parent.mkdir(parents=True, exist_ok=True)
predictions_df.to_csv(predictions_path, index=False)
print(f"Saved predictions to: {predictions_path}")
print("\nPipeline execution completed successfully!")