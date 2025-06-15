import pandas as pd
import numpy as np
import csv
from datetime import datetime
from tqdm import tqdm
import os

def safe_convert_float(x):
    """Safe conversion to float with European number format support"""
    try:
        if pd.isna(x) or str(x).strip() in ['', 'NA', 'nan', 'null']:
            return np.nan
        # Handle European-style numbers (1.234,56 -> 1234.56)
        x = str(x).replace('.', '').replace(',', '.')
        return float(x)
    except:
        return np.nan

def safe_convert_int(x):
    """Safe conversion to integer"""
    try:
        if pd.isna(x) or str(x).strip() in ['', 'NA', 'nan', 'null']:
            return np.nan
        # Handle European-style numbers
        x = str(x).replace('.', '').replace(',', '.')
        return int(float(x))  # Convert to float first to handle decimals
    except:
        return np.nan

def validate_financial_values(df):
    """Fix extreme values and invalid claims"""
    # Convert currency values to reasonable scale
    currency_cols = ['suminsured', 'totalpremium', 'totalclaims', 'customvalueestimate']
    for col in currency_cols:
        if col in df.columns:
            # Remove currency symbols if present
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
            
            # Convert to thousands or millions if needed
            if df[col].mean() > 1e6:
                df[col] = df[col] / 1e6
                print(f"Converted {col} to millions")
    
    # Handle zero claims - replace with small values if needed
    if 'totalclaims' in df.columns:
        claim_mask = df['totalclaims'] <= 0
        if claim_mask.any():
            df.loc[claim_mask, 'totalclaims'] = df['totalpremium'] * 0.01  # 1% of premium
    
    return df

def load_and_preprocess_data(file_path):
    """Load and preprocess insurance data with robust error handling"""
    try:
        print("⏳ Loading and preprocessing data...")
        
        # Initialize data containers
        data = {
            'policyid': [], 'gender': [], 'country': [], 'province': [],
            'postalcode': [], 'vehicletype': [], 'registrationyear': [],
            'make': [], 'model': [], 'suminsured': [], 'totalpremium': [],
            'totalclaims': [], 'customvalueestimate': []
        }
        
        # Process file line by line
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            reader = csv.reader(f, delimiter='|')
            header = [col.strip().lower() for col in next(reader)]
            
            # Get column indices
            col_indices = {col: header.index(col) if col in header else -1 for col in data.keys()}
            
            # Process all rows
            for row in tqdm(reader, desc="Processing rows"):
                try:
                    for col in data.keys():
                        if col_indices[col] != -1:
                            val = row[col_indices[col]]
                            if col in ['suminsured', 'totalpremium', 'totalclaims', 'customvalueestimate']:
                                data[col].append(safe_convert_float(val))
                            elif col == 'registrationyear':
                                data[col].append(safe_convert_int(val))
                            else:
                                data[col].append(val.strip() if val else np.nan)
                except Exception as e:
                    continue
        
        # Create DataFrame
        df = pd.DataFrame({k: v for k, v in data.items() if len(v) > 0})
        
        if df.empty:
            raise ValueError("No data was loaded - empty DataFrame")
        
        # Data Quality Report
        print("\n🔍 Data Quality Report:")
        print(f"Initial Records: {len(df):,}")
        print("\nMissing Values:")
        print(df.isnull().sum())
        print(f"\nDuplicate Records: {df.duplicated().sum()}")
        
        # Handle missing values
        if 'customvalueestimate' in df.columns and 'suminsured' in df.columns:
            df['customvalueestimate'] = df['customvalueestimate'].fillna(df['suminsured'])
        
        # Drop rows with missing essential values
        essential_cols = ['totalpremium', 'totalclaims']
        df = df.dropna(subset=[col for col in essential_cols if col in df.columns])
        
        # Feature Engineering
        current_year = datetime.now().year
        if 'registrationyear' in df.columns:
            df['vehicle_age'] = current_year - df['registrationyear']
            # Create age groups safely
            try:
                df['age_group'] = pd.cut(
                    df['vehicle_age'],
                    bins=[0, 5, 10, 15, 20, 100],
                    labels=['0-5', '6-10', '11-15', '16-20', '20+'],
                    right=False
                )
            except Exception as e:
                print(f"⚠️ Could not create age groups: {str(e)}")
        
        if 'totalpremium' in df.columns and 'totalclaims' in df.columns:
            df['loss_ratio'] = np.where(
                df['totalpremium'] > 0,
                df['totalclaims'] / df['totalpremium'],
                np.nan
            )
            df['profit_margin'] = df['totalpremium'] - df['totalclaims']
        
        # Validate financial values
        df = validate_financial_values(df)
        
        # Optimize data types
        numeric_cols = ['suminsured', 'totalpremium', 'totalclaims', 
                       'customvalueestimate', 'vehicle_age', 'loss_ratio',
                       'profit_margin']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        
        categorical_cols = ['gender', 'country', 'province', 'vehicletype', 
                           'make', 'model', 'age_group']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        print(f"\n✅ Final dataset: {len(df):,} records")
        print(f"📊 Memory usage: {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
        
        return df
    
    except Exception as e:
        print(f"❌ Error in data loading: {str(e)}")
        return None