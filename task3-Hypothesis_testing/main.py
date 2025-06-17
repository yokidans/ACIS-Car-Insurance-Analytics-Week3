import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import psutil
import sys

# Import our modules
from src.data_processing import load_and_preprocess_data
from src.eda import perform_comprehensive_eda
from src.hypothesis_testing import perform_comprehensive_hypothesis_tests
from src.reporting import generate_comprehensive_insights
from src.modeling import build_ml_models, build_optimal_premium_model

def main():
    print("=== ACIS CAR INSURANCE ANALYTICS PLATFORM ===")
    print("=== COMPREHENSIVE RISK AND PRICING ANALYSIS ===\n")
    
    # Configuration
    data_file = 'data/raw/insurance_claims.csv'
    output_dir = 'reports'
    processed_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Phase 1: Data Loading & Preprocessing
    print("\n" + "="*50)
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    print("="*50)
    df = load_and_preprocess_data(data_file)
    if df is None:
        print("❌ Failed to load data. Exiting.")
        sys.exit(1)
    
    # Save processed data
    processed_file = os.path.join(processed_dir, 'processed_insurance_data.csv')
    df.to_csv(processed_file, index=False)
    print(f"\n✅ Processed data saved to {processed_file}")
    
    # Phase 2: Exploratory Data Analysis
    print("\n" + "="*50)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("="*50)
    if not perform_comprehensive_eda(df):
        print("❌ EDA failed. Continuing with analysis but some insights may be missing.")
    
    # Memory check and sampling configuration
    print(f"\n🖥️ Available RAM: {psutil.virtual_memory().available/1e9:.1f} GB")
    if psutil.virtual_memory().available < 8e9:  # Less than 8GB
        SAMPLE_SIZE = 15000
    else:
        SAMPLE_SIZE = 30000
    print(f"Using sample size: {SAMPLE_SIZE:,} records")
    
    # Phase 3: Hypothesis Testing
    print("\n" + "="*50)
    print("PHASE 3: HYPOTHESIS TESTING")
    print("="*50)
    if not perform_comprehensive_hypothesis_tests(df, sample_size=SAMPLE_SIZE):
        print("❌ Hypothesis testing failed. Continuing with analysis but some tests may be missing.")
    
    # Phase 4: Machine Learning Modeling
    print("\n" + "="*50)
    print("PHASE 4: MACHINE LEARNING MODELING")
    print("="*50)
    try:
        model_results = build_ml_models(df, sample_size=SAMPLE_SIZE)
        if model_results is None:
            raise ValueError("Model building returned no results")
            
        # Print model performance summary
        print("\n⭐ Model Performance Summary:")
        for name, res in model_results.items():
            print(f"{name}: MAE = {res['mae']:.2f} | R2 = {res['r2']:.2f} | Time = {res['time']:.1f}s")
            
    except Exception as e:
        print(f"❌ Critical error in modeling phase: {str(e)}")
        model_results = None
    
    # Premium Optimization Model
    print("\n" + "="*50)
    print("PREMIUM OPTIMIZATION MODELING")
    print("="*50)
    try:
        premium_model = build_optimal_premium_model(df, sample_size=SAMPLE_SIZE)
        if premium_model:
            print(f"\nPremium Model Performance:")
            print(f"- MAE: {premium_model['mae']:.2f}")
            print(f"- RMSE: {premium_model['rmse']:.2f}")
            print(f"- R2: {premium_model['r2']:.2f}")
    except Exception as e:
        print(f"❌ Error in premium modeling: {str(e)}")
        premium_model = None
    
    # Phase 5: Reporting & Insights
    print("\n" + "="*50)
    print("PHASE 5: REPORTING & INSIGHTS")
    print("="*50)
    if not generate_comprehensive_insights(df, model_results, premium_model):
        print("❌ Insight generation failed. Some recommendations may be missing.")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    # Set display options
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    
    # Visualization settings
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Execute main function
    main()