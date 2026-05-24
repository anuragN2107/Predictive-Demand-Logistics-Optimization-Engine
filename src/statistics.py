# ==============================================================================
# MODULE 2: INTEGRITY & HYPOTHESIS TESTING ENGINE
# ==============================================================================

import scipy.stats as stats

def run_statistical_validation(df):
    print("Initiating mathematical verification of data properties...")
    results = {}
    
    # 1. Distribution Testing (Kolmogorov-Smirnov Test)
    std_demand = (df['demand_count'] - df['demand_count'].mean()) / df['demand_count'].std()
    ks_stat, ks_p = stats.kstest(std_demand, 'norm')
    results['ks_p_value'] = ks_p
    
    # 2. Non-Parametric A/B Hypothesis Testing (Weekday vs Weekend)
    weekday_df = df[df['is_weekend'] == 0]['demand_count']
    weekend_df = df[df['is_weekend'] == 1]['demand_count']
    u_stat, mw_p = stats.mannwhitneyu(weekday_df, weekend_df, alternative='two-sided')
    results['mann_whitney_p_value'] = mw_p
    
    print(f"   - Distribution KS p-value: {ks_p:.4e}")
    print(f"   - Operational A/B Shift p-value: {mw_p:.4e}")
    return results
