# ==============================================================================
# MODULE 3: MACHINE LEARNING CLASSIFICATION TOURNAMENT
# ==============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb

def run_classification_tournament(df):
    print("🏋️ Initializing Machine Learning Classifier Tournament...")
    
    feature_cols = ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 
                    'demand_lag_1h', 'demand_lag_2h', 'demand_lag_24h', 
                    'rolling_mean_3h', 'rolling_std_3h']
    
    X = df[feature_cols].values
    y = df['high_demand_target'].values
    
    # Chronological Split (80/20 Rule)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale Features (Averting data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost Classifier": xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42)
    }
    
    performance_log = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        performance_log[name] = {
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0)
        }
        
    print("✅ Classification Tournament Completed.")
    return performance_log
