def generate_comprehensive_insights(df, model_results=None, premium_model=None):
    """Generate comprehensive business insights with recommendations"""
    try:
        print("\n📈 Generating Comprehensive Business Insights...")
        
        # 1. Executive Summary
        print("\n=== EXECUTIVE SUMMARY ===")
        avg_loss_ratio = df['loss_ratio'].mean()
        avg_profit_margin = df['profit_margin'].mean()
        total_premium = df['totalpremium'].sum()
        total_claims = df['totalclaims'].sum()
        
        print(f"\nOverall Portfolio Metrics:")
        print(f"- Average Loss Ratio: {avg_loss_ratio:.2%}")
        print(f"- Average Profit Margin: ${avg_profit_margin:,.2f}")
        print(f"- Total Premium Volume: ${total_premium:,.2f}")
        print(f"- Total Claims Paid: ${total_claims:,.2f}")
        print(f"- Policies Analyzed: {len(df):,}")
        
        # 2. Risk Analysis
        print("\n=== RISK ANALYSIS ===")
        generate_risk_insights(df)
        
        # 3. Profitability Analysis
        print("\n=== PROFITABILITY ANALYSIS ===")
        generate_profitability_insights(df)
        
        # 4. Modeling Insights
        if model_results:
            print("\n=== MODELING INSIGHTS ===")
            generate_modeling_insights(model_results)
        
        # 5. Premium Optimization Insights
        if premium_model:
            print("\n=== PREMIUM OPTIMIZATION INSIGHTS ===")
            generate_premium_insights(premium_model)
        
        # 6. Actionable Recommendations
        print("\n=== ACTIONABLE RECOMMENDATIONS ===")
        generate_recommendations(df, model_results, premium_model)
        
        return True
    
    except Exception as e:
        print(f"❌ Error in generating insights: {str(e)}")
        return False

def generate_risk_insights(df):
    """Generate insights about risk factors"""
    # High risk provinces
    if 'province' in df.columns and 'loss_ratio' in df.columns:
        high_risk_provinces = df.groupby('province', observed=True)['loss_ratio'].mean()\
                              .nlargest(3).index.tolist()
        print(f"\nHighest Risk Provinces:")
        for i, province in enumerate(high_risk_provinces, 1):
            lr = df[df['province'] == province]['loss_ratio'].mean()
            print(f"{i}. {province}: {lr:.2%} loss ratio")
    
    # High risk vehicle types
    if 'vehicletype' in df.columns and 'loss_ratio' in df.columns:
        high_risk_vehicles = df.groupby('vehicletype', observed=True)['loss_ratio'].mean()\
                             .nlargest(2).index.tolist()
        print(f"\nHighest Risk Vehicle Types:")
        for i, vehicle in enumerate(high_risk_vehicles, 1):
            lr = df[df['vehicletype'] == vehicle]['loss_ratio'].mean()
            print(f"{i}. {vehicle}: {lr:.2%} loss ratio")
    
    # Risk by age group
    if 'age_group' in df.columns and 'loss_ratio' in df.columns:
        print("\nLoss Ratio by Vehicle Age Group:")
        age_stats = df.groupby('age_group', observed=True)['loss_ratio'].mean()\
                    .sort_values(ascending=False)
        for age_group, lr in age_stats.items():
            print(f"- {age_group} years: {lr:.2%}")

def generate_profitability_insights(df):
    """Generate insights about profitability factors"""
    # Most profitable provinces
    if 'province' in df.columns and 'profit_margin' in df.columns:
        profitable_provinces = df.groupby('province', observed=True)['profit_margin'].mean()\
                               .nlargest(3).index.tolist()
        print(f"\nMost Profitable Provinces:")
        for i, province in enumerate(profitable_provinces, 1):
            pm = df[df['province'] == province]['profit_margin'].mean()
            print(f"{i}. {province}: ${pm:,.2f} average profit margin")
    
    # Most profitable vehicle types
    if 'vehicletype' in df.columns and 'profit_margin' in df.columns:
        profitable_vehicles = df.groupby('vehicletype', observed=True)['profit_margin'].mean()\
                             .nlargest(2).index.tolist()
        print(f"\nMost Profitable Vehicle Types:")
        for i, vehicle in enumerate(profitable_vehicles, 1):
            pm = df[df['vehicletype'] == vehicle]['profit_margin'].mean()
            print(f"{i}. {vehicle}: ${pm:,.2f} average profit margin")
    
    # Profitability by gender
    if 'gender' in df.columns and 'profit_margin' in df.columns:
        print("\nProfit Margin by Gender:")
        gender_stats = df.groupby('gender', observed=True)['profit_margin'].mean()\
                       .sort_values(ascending=False)
        for gender, pm in gender_stats.items():
            print(f"- {gender}: ${pm:,.2f}")

def generate_modeling_insights(model_results):
    """Generate insights from modeling results"""
    best_model_name = max(model_results, key=lambda x: model_results[x]['R2'])
    best_r2 = model_results[best_model_name]['R2']
    
    print(f"\nBest Performing Model: {best_model_name} (R2: {best_r2:.2f})")
    
    # Feature importance insights
    if 'Random Forest' in model_results:
        rf_importances = pd.DataFrame({
            'Feature': model_results['Random Forest']['feature_names'],
            'Importance': model_results['Random Forest']['model'].named_steps['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        top_features = rf_importances.head(5)['Feature'].tolist()
        print("\nTop 5 Predictive Features (Random Forest):")
        for i, feature in enumerate(top_features, 1):
            print(f"{i}. {feature}")

def generate_premium_insights(premium_model):
    """Generate insights from premium optimization model"""
    print("\nKey Factors Influencing Optimal Premiums:")
    # Extract feature importances from the model
    # (Implementation would be similar to generate_modeling_insights)
    
    print("\nPremium Optimization Opportunities:")
    print("- Adjust premiums based on vehicle type and age")
    print("- Implement location-based pricing strategies")
    print("- Offer discounts for low-risk customer segments")

def generate_recommendations(df, model_results=None, premium_model=None):
    """Generate actionable business recommendations"""
    print("\n1. Pricing Strategy Recommendations:")
    print("- Implement risk-based pricing for high-risk segments")
    print("- Offer competitive premiums for low-risk segments")
    print("- Introduce graduated pricing based on vehicle age")
    
    print("\n2. Marketing Strategy Recommendations:")
    print("- Target low-risk, high-profit customer segments")
    print("- Develop retention programs for profitable customers")
    print("- Create tailored campaigns for specific vehicle types")
    
    print("\n3. Risk Management Recommendations:")
    print("- Implement stricter underwriting for high-risk categories")
    print("- Consider partnerships with repair shops in high-risk areas")
    print("- Develop driver education programs for high-risk regions")
    
    if model_results:
        print("\n4. Data-Driven Decision Making Recommendations:")
        print("- Incorporate predictive modeling into underwriting")
        print("- Use model insights to refine risk assessment")
        print("- Continuously monitor model performance and update regularly")
    
    if premium_model:
        print("\n5. Premium Optimization Recommendations:")
        print("- Implement model-based premium calculations for new policies")
        print("- Test price sensitivity in different segments")
        print("- Monitor impact of premium changes on portfolio mix")