# Predictive Demand & Logistics Optimization Engine 🚖

An end-to-end data architecture designed to ingest real-world, large-scale transaction networks, perform robust statistical validation, classify operational risks, and generate continuous chronological forecasts.

## 🚀 Core Objective & Use Cases
This engine processes raw transactional databases into chronological, hourly time-series blocks to solve structural supply-demand mismatches. 
* **Dynamic Fleet Rebalancing:** Pre-positioning logistics assets in high-demand zones before requests spike.
* **Supply-Chain Micro-Fulfillment:** Managing hyper-local inventory staging to eliminate stockouts.
* **Dynamic Pricing Models:** Aligning demand forecasts with revenue management algorithms.

## 🛠️ Tech Stack & Ecosystem
* **Data Wrangling & Pipeline:** `Pandas`, `NumPy`
* **Statistical Integrity Layer:** `SciPy (Stats)`, `Statsmodels`
* **Machine Learning Tournament:** `Scikit-Learn`, `XGBoost`
* **Visualization Engine:** `Seaborn`, `Matplotlib`

## 🔬 Phase 1: Mathematical Validation Checks
Before training, the data properties were evaluated to confirm modeling assumptions:
* **Kolmogorov-Smirnov (KS) Test:** Evaluated the continuous demand distribution against a standard normal distribution baseline ($D = sup | F_n(x) - F(x) |$). A $p$-value $< 0.05$ rejected normality, validating the need for non-linear machine learning architectures.
* **Mann-Whitney U Test:** Run as a non-parametric hypothesis check across operational days. It statistically proved that weekend demand profiles shift significantly compared to weekdays ($p < 0.05$).

## 📊 Phase 2: Machine Learning Tournament Performance
The system executed a tournament across 5 distinct classification models to sort high-demand spikes (>75th percentile):

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* |
| **K-Nearest Neighbors** | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* |
| **Support Vector Machine** | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* |
| **Random Forest** | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* |
| **XGBoost Classifier** | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* | *0.XXXX* |

*(Note: In production environments, we optimize for **Precision** to avoid wasting fleet resources on false positives, while balancing **Recall** to protect revenue).*

## 📈 Phase 3: Continuous Time-Series Forecasting
After verifying data stationarity using the **Augmented Dickey-Fuller (ADF) Test**, we implemented a dual-track forecasting strategy to project continuous volume request horizons:
1. **Statistical Track (SARIMAX):** Implemented a $SARIMAX(1,0,1) \times (1,0,0)_{24}$ process to model structural 24-hour daily seasonality rhythms.
2. **Machine Learning Track (XGBoost Regressor):** Formatted sequential time indexes into structured tabular lags ($t-1, t-2, t-24$) and rolling windows to adapt quickly to abrupt trend shifts.

### Forecasting Error Matrix Summary
* **SARIMAX:** MAE: *XX.XX* | RMSE: *XX.XX*
* **XGBoost Regressor:** MAE: *XX.XX* | RMSE: *XX.XX*

## 🔮 Future Engineering Roadmap (MLOps Integration)
To transition this project into a scalable production layer, the next development sprint will focus on:
1. **Experiment Tracking:** Integrating `MLflow` to register hyperparameters, model weights, and evaluation metrics dynamically.
2. **Containerization:** Authoring a `Dockerfile` to completely isolate dependency version profiles across local and cloud environments.
3. **CI/CD Automation:** Deploying `GitHub Actions` to run automated code quality tests (`pytest`, `flake8`) on incoming pipeline patches.
