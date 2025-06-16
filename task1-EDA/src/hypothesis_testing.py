# src/hypothesis_testing.py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm
import warnings

def perform_comprehensive_hypothesis_tests(df, sample_size=20000, random_state=42):
    """Optimized hypothesis testing with sampling"""
    try:
        print("\n🔬 Performing Comprehensive Hypothesis Tests...")
        
        # Create representative sample if needed
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=random_state)
            print(f"Using sampled data ({sample_size:,} records)")
        
        # 1. Province Risk Differences
        if 'province' in df and 'loss_ratio' in df:
            print("\n=== PROVINCE RISK DIFFERENCES ===")
            provinces = df['province'].value_counts().nlargest(5).index
            province_data = []
            
            for prov in provinces:
                prov_sample = df[df['province'] == prov]['loss_ratio'].dropna()
                if len(prov_sample) > 500:  # Downsample large groups
                    prov_sample = prov_sample.sample(500, random_state=random_state)
                province_data.append(prov_sample)
            
            # Kruskal-Wallis test
            h_stat, p_val = stats.kruskal(*province_data)
            print(f"Kruskal-Wallis H: {h_stat:.2f}, p-value: {p_val:.4f}")
        
        # 2. Gender Differences
        if 'gender' in df and 'loss_ratio' in df:
            print("\n=== GENDER RISK DIFFERENCES ===")
            genders = df['gender'].value_counts().nlargest(2).index
            
            if len(genders) == 2:
                group1 = df[df['gender'] == genders[0]]['loss_ratio'].dropna()
                group2 = df[df['gender'] == genders[1]]['loss_ratio'].dropna()
                
                # Mann-Whitney U test
                u_stat, p_val = stats.mannwhitneyu(group1, group2)
                print(f"Mann-Whitney U: {u_stat:.2f}, p-value: {p_val:.4f}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error in hypothesis testing: {str(e)}")
        return False