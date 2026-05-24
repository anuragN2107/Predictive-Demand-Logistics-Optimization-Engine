# ==============================================================================
# MODULE 1: DATA INGESTION 
# ==============================================================================

import pandas as pd
import numpy as np

def load_clean_and_engineer_data(url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"):
    print("Ingesting production Parquet files directly from cloud infrastructure...")
    df_raw = pd.read_parquet(url)
    
    # Subsetting columns to optimize memory footprint
    keep_cols = ['tpep_pickup_datetime', 'trip_distance', 'fare_amount', 'tip_amount']
    df = df_raw[keep_cols].copy()
    df.rename(columns={'tpep_pickup_datetime': 'pickup_time'}, inplace=True)
    
    # Data Cleaning & Structural Time-Series Aggregation
    df['pickup_time'] = pd.to_datetime(df['pickup_time'])
    df = df[(df['pickup_time'] >= '2023-01-01') & (df['pickup_time'] < '2023-02-01')]
    df['hourly_window'] = df['pickup_time'].dt.floor('h')
    
    hourly_df = df.groupby('hourly_window').agg(
        demand_count=('fare_amount', 'count'),
        avg_distance=('trip_distance', 'mean'),
        total_revenue=('fare_amount', 'sum'),
        avg_tip=('tip_amount', 'mean')
    ).reset_index()
    
    # Ensure chronological continuity across the entire month
    full_time_range = pd.date_range(start='2023-01-01 00:00:00', end='2023-01-31 23:00:00', freq='h')
    hourly_df = hourly_df.set_index('hourly_window').reindex(full_time_range, fill_value=0).reset_index()
    hourly_df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # Advanced Feature Engineering
    hourly_df['hour'] = hourly_df['timestamp'].dt.hour
    hourly_df['day_of_week'] = hourly_df['timestamp'].dt.dayofweek
    hourly_df['is_weekend'] = hourly_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Cyclical Encoding
    hourly_df['hour_sin'] = np.sin(2 * np.pi * hourly_df['hour'] / 24.0)
    hourly_df['hour_cos'] = np.cos(2 * np.pi * hourly_df['hour'] / 24.0)
    
    # Chronological Momentum Lags & Rolling Statistics
    hourly_df['demand_lag_1h'] = hourly_df['demand_count'].shift(1)
    hourly_df['demand_lag_2h'] = hourly_df['demand_count'].shift(2)
    hourly_df['demand_lag_24h'] = hourly_df['demand_count'].shift(24)
    hourly_df['rolling_mean_3h'] = hourly_df['demand_count'].shift(1).rolling(window=3).mean()
    hourly_df['rolling_std_3h'] = hourly_df['demand_count'].shift(1).rolling(window=3).std()
    
    hourly_df.dropna(inplace=True)
    
    # Define Classification Binary Target (>75th percentile)
    threshold = hourly_df['demand_count'].quantile(0.75)
    hourly_df['high_demand_target'] = (hourly_df['demand_count'] > threshold).astype(int)
    
    print(f"Data pipeline processed {len(hourly_df)} chronological records successfully.")
    return hourly_df, threshold
