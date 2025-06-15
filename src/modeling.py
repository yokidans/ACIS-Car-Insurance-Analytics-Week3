import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm

def build_ml_models(df, target='totalclaims', test_size=0.2, random_state=42):
    """Build and evaluate machine learning models"""
    try:
        print("\n🤖 Building Machine Learning Models...")
        
        # Feature selection
        features = [
            'vehicletype', 'vehicle_age', 'suminsured', 
            'province', 'gender', 'make'
        ]
        features = [f for f in features if f in df.columns]
        
        if not features or target not in df.columns:
            print("❌ Required features missing for modeling")
            return None
            
        X = df[features]
        y = df[target]
        
        # Preprocessing
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Model configurations
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                n_jobs=-1,
                random_state=random_state
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=random_state
            )
        }
        
        results = {}
        for name, model in tqdm(models.items(), desc="Training models"):
            start_time = time.time()
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            results[name] = {
                'model': pipeline,
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'time': time.time() - start_time
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Error in model building: {str(e)}")
        return None

def build_optimal_premium_model(df, target_loss_ratio=0.6, test_size=0.2, random_state=42):
    """Build model for premium optimization"""
    try:
        print("\n💰 Building Optimal Premium Model...")
        
        df['target_premium'] = df['totalclaims'] / target_loss_ratio
        
        features = [
            'vehicletype', 'vehicle_age', 'suminsured', 
            'province', 'gender', 'make'
        ]
        features = [f for f in features if f in df.columns]
        
        X = df[features]
        y = df['target_premium']
        
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=random_state
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        return {
            'pipeline': pipeline,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
    except Exception as e:
        print(f"❌ Error in premium modeling: {str(e)}")
        return None