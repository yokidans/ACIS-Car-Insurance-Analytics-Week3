# Machine Learning for Insurance Risk Prediction

## Overview
This project implements a machine learning pipeline to predict insurance risk using processed insurance data. It includes data preparation, preprocessing, model training, evaluation, and feature importance analysis.

## Key Features
- **End-to-End ML Pipeline**: Covers data preparation, feature engineering, model training, evaluation, and interpretation.
- **Advanced Techniques**: Utilizes SHAP values for interpretability, stratified sampling, and hyperparameter tuning.
- **Business-Ready Outputs**: Provides risk segmentation, premium recommendations, and visual explanations of model decisions.
- **Reproducibility**: Ensures consistency with fixed random states, pipeline encapsulation, and versioned outputs.

## Models Evaluated
1. **Linear Regression**
2. **Random Forest Regressor**
3. **XGBoost Regressor**

## Results
- **Linear Regression**: MAE: 80.54M, RMSE: 6.48B, R²: 0.0001  
- **Random Forest**: MAE: 97.41M, RMSE: 7.22B, R²: -0.075  
- **XGBoost**: MAE: 878.09M, RMSE: 6.97B, R²: -0.008  

## Top Features (Random Forest)
1. `suminsured`  
2. `vehicle_age`  
3. `province_Western Cape`  

## Dependencies
- Core Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`  
- ML Tools: `scikit-learn`, `XGBoost`, `SHAP`  

## Usage
1. Clone the repository.  
2. Install dependencies: `pip install -r requirements.txt`.  
3. Run the notebook: `03_modeling.ipynb`.  

## Outputs
- Feature importance plots (`./reports/figures/`).  
- SHAP value visualizations (`./reports/figures/`).  
