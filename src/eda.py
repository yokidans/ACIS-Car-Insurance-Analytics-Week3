import os  # Add this import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def perform_comprehensive_eda(df, output_dir='reports/figures'):
    """Enhanced EDA with more visualizations and statistical analysis"""
    try:
        os.makedirs(output_dir, exist_ok=True)  # Now has access to 'os'
        print("\n📊 Performing Comprehensive Exploratory Data Analysis...")
        
        # 1. Data Structure Analysis
        print("\n=== DATA STRUCTURE ===")
        print(df.info())
        
        # 2. Financial Metrics Analysis
        analyze_financial_metrics(df, output_dir)
        
        # 3. Geographic Analysis
        analyze_geographic_data(df, output_dir)
        
        # 4. Vehicle Characteristics Analysis
        analyze_vehicle_data(df, output_dir)
        
        # 5. Demographic Analysis
        analyze_demographic_data(df, output_dir)
        
        # 6. Correlation Analysis
        analyze_correlations(df, output_dir)
        
        # 7. Temporal Analysis (if date column exists)
        if 'transactiondate' in df.columns:
            analyze_temporal_trends(df, output_dir)
        
        return True
    
    except Exception as e:
        print(f"❌ Error in EDA: {str(e)}")
        return False

def analyze_financial_metrics(df, output_dir):
    """Analyze premium, claims, and financial ratios"""
    print("\n=== FINANCIAL METRICS ANALYSIS ===")
    
    plt.figure(figsize=(18, 12))
    
    # Premium distribution with log scale
    plt.subplot(2, 3, 1)
    sns.histplot(np.log1p(df['totalpremium']), bins=50, kde=True)
    plt.title('Log Premium Distribution')
    
    # Claims distribution with log scale
    plt.subplot(2, 3, 2)
    sns.histplot(np.log1p(df['totalclaims']), bins=50, kde=True)
    plt.title('Log Claims Distribution')
    
    # Loss ratio distribution
    plt.subplot(2, 3, 3)
    sns.histplot(df['loss_ratio'], bins=50, kde=True)
    plt.title('Loss Ratio Distribution')
    
    # Profit margin distribution
    plt.subplot(2, 3, 4)
    sns.histplot(df['profit_margin'], bins=50, kde=True)
    plt.title('Profit Margin Distribution')
    
    # Premium vs Claims scatter
    plt.subplot(2, 3, 5)
    sns.scatterplot(x='totalpremium', y='totalclaims', data=df, alpha=0.3)
    plt.title('Premium vs Claims')
    
    # Boxplot of loss ratio by age group
    plt.subplot(2, 3, 6)
    sns.boxplot(x='age_group', y='loss_ratio', data=df)
    plt.title('Loss Ratio by Vehicle Age Group')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/financial_metrics.png')
    plt.show()

def analyze_geographic_data(df, output_dir):
    """Analyze geographic patterns in risk and profitability"""
    if 'province' not in df.columns:
        return
    
    print("\n=== GEOGRAPHIC ANALYSIS ===")
    
    # Calculate province metrics
    province_stats = df.groupby('province', observed=True).agg({
        'totalpremium': ['count', 'mean', 'sum'],
        'totalclaims': ['mean', 'sum'],
        'loss_ratio': 'mean',
        'profit_margin': 'mean'
    }).sort_values(('loss_ratio', 'mean'), ascending=False)
    
    print("\nProvince Statistics:")
    print(province_stats)
    
    # Top and bottom provinces by loss ratio
    top_provinces = province_stats.nlargest(5, ('loss_ratio', 'mean'))
    bottom_provinces = province_stats.nsmallest(5, ('loss_ratio', 'mean'))
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=top_provinces[('loss_ratio', 'mean')], 
                y=top_provinces.index.get_level_values(0))
    plt.title('Top 5 High Risk Provinces by Loss Ratio')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=bottom_provinces[('loss_ratio', 'mean')], 
                y=bottom_provinces.index.get_level_values(0))
    plt.title('Top 5 Low Risk Provinces by Loss Ratio')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/province_risk.png')
    plt.show()

def analyze_vehicle_data(df, output_dir):
    """Analyze vehicle-related patterns"""
    if 'vehicletype' not in df.columns:
        return
    
    print("\n=== VEHICLE ANALYSIS ===")
    
    # Vehicle type analysis
    vehicle_stats = df.groupby('vehicletype', observed=True).agg({
        'totalpremium': ['count', 'mean'],
        'totalclaims': 'mean',
        'loss_ratio': 'mean',
        'profit_margin': 'mean'
    }).sort_values(('loss_ratio', 'mean'), ascending=False)
    
    print("\nVehicle Type Statistics:")
    print(vehicle_stats)
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Loss ratio by vehicle type
    plt.subplot(2, 2, 1)
    sns.barplot(x='loss_ratio', y='vehicletype', 
                data=df, estimator=np.mean, ci=None)
    plt.title('Average Loss Ratio by Vehicle Type')
    
    # Premium distribution by vehicle type
    plt.subplot(2, 2, 2)
    sns.boxplot(x='vehicletype', y='totalpremium', data=df)
    plt.xticks(rotation=45)
    plt.title('Premium Distribution by Vehicle Type')
    
    # Vehicle age vs loss ratio
    plt.subplot(2, 2, 3)
    sns.regplot(x='vehicle_age', y='loss_ratio', data=df, 
                scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
    plt.title('Vehicle Age vs Loss Ratio')
    
    # Make analysis (top 10 by count)
    if 'make' in df.columns:
        top_makes = df['make'].value_counts().nlargest(10).index
        make_stats = df[df['make'].isin(top_makes)].groupby('make', observed=True)['loss_ratio'].mean()
        
        plt.subplot(2, 2, 4)
        sns.barplot(x=make_stats.values, y=make_stats.index)
        plt.title('Loss Ratio by Top 10 Vehicle Makes')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vehicle_analysis.png')
    plt.show()

def analyze_demographic_data(df, output_dir):
    """Analyze demographic patterns (gender, etc.)"""
    if 'gender' not in df.columns:
        return
    
    print("\n=== DEMOGRAPHIC ANALYSIS ===")
    
    # Gender analysis
    gender_stats = df.groupby('gender', observed=True).agg({
        'totalpremium': ['count', 'mean'],
        'totalclaims': 'mean',
        'loss_ratio': 'mean',
        'profit_margin': 'mean'
    })
    
    print("\nGender Statistics:")
    print(gender_stats)
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.barplot(x='gender', y='loss_ratio', data=df, estimator=np.mean)
    plt.title('Loss Ratio by Gender')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='gender', y='totalpremium', data=df)
    plt.title('Premium Distribution by Gender')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(x='gender', y='totalclaims', data=df)
    plt.title('Claims Distribution by Gender')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demographic_analysis.png')
    plt.show()

def analyze_correlations(df, output_dir):
    """Analyze correlations between numeric variables"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return
    
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Correlation matrix
    corr_matrix = numeric_df.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
    plt.title('Correlation Matrix (Numeric Variables)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.show()
    
    # Pairplot of key variables
    key_vars = ['totalpremium', 'totalclaims', 'loss_ratio', 
               'profit_margin', 'vehicle_age']
    key_vars = [v for v in key_vars if v in numeric_df.columns]
    
    if len(key_vars) > 1:
        sns.pairplot(df[key_vars], corner=True)
        plt.suptitle('Pairwise Relationships of Key Variables', y=1.02)
        plt.savefig(f'{output_dir}/pairplot.png')
        plt.show()

def analyze_temporal_trends(df, output_dir):
    """Analyze temporal patterns if date column exists"""
    print("\n=== TEMPORAL ANALYSIS ===")
    
    # Convert to datetime and extract features
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    df['transaction_month'] = df['transactiondate'].dt.to_period('M')
    
    # Monthly aggregates
    monthly_stats = df.groupby('transaction_month').agg({
        'totalpremium': 'sum',
        'totalclaims': 'sum',
        'policyid': 'count'
    }).reset_index()
    
    monthly_stats['transaction_month'] = monthly_stats['transaction_month'].astype(str)
    monthly_stats['loss_ratio'] = monthly_stats['totalclaims'] / monthly_stats['totalpremium']
    
    print("\nMonthly Statistics:")
    print(monthly_stats)
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Premium and claims over time
    plt.subplot(2, 2, 1)
    sns.lineplot(x='transaction_month', y='totalpremium', data=monthly_stats, label='Premium')
    sns.lineplot(x='transaction_month', y='totalclaims', data=monthly_stats, label='Claims')
    plt.xticks(rotation=45)
    plt.title('Monthly Premium and Claims')
    plt.legend()
    
    # Loss ratio over time
    plt.subplot(2, 2, 2)
    sns.lineplot(x='transaction_month', y='loss_ratio', data=monthly_stats)
    plt.xticks(rotation=45)
    plt.title('Monthly Loss Ratio')
    
    # Policy count over time
    plt.subplot(2, 2, 3)
    sns.lineplot(x='transaction_month', y='policyid', data=monthly_stats)
    plt.xticks(rotation=45)
    plt.title('Monthly Policy Count')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_analysis.png')
    plt.show()