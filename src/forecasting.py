# ==============================================================================
# MODULE 4: ADVANCED CONTINUOUS TIME-SERIES FORECASTING ENGINE
# ==============================================================================

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import numpy as np

def run_forecasting_engine(df):
    print("📈 Executing continuous forecasting tracking modules...")
    
    # 1. Stationarity Check (ADF Test)
    adf_test = adfuller(df['demand_count'].values)
    is_stationary = adf_test[1] < 0.05
    
    # Chronological Split
    train_size = int(len(df) * 0.80)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # 2. Track A: Statistical Seasonal Forecast (SARIMAX)
    sarima = SARIMAX(train_df['demand_count'].values, order=(1,0,1), seasonal_order=(1,0,0,24))
    sarima_fit = sarima.fit(disp=False)
    sarima_preds = sarima_fit.forecast(steps=len(test_df))
    
    # 3. Track B: Machine Learning Forecast (XGBoost Regressor)
    reg_features = ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 
                    'demand_lag_1h', 'demand_lag_2h', 'demand_lag_24h', 
                    'rolling_mean_3h', 'rolling_std_3h']
    
    X_train, y_train = train_df[reg_features].values, train_df['demand_count'].values
    X_test, y_test = test_df[reg_features].values, test_df['demand_count'].values
    
    xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_reg.fit(X_train, y_train)
    xgb_preds = xgb_reg.predict(X_test)
    
    # Package Performance Errors
    metrics = {
        "SARIMAX": {
            "MAE": mean_absolute_error(y_test, sarima_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, sarima_preds))
        },
        "XGBoost Regressor": {
            "MAE": mean_absolute_error(y_test, xgb_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, xgb_preds))
        }
    }
    
    print("✅ Forecasting pipelines run complete.")
    return metrics, is_stationary
