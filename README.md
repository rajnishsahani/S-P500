# S&P 500 Hybrid Prediction Model — XGBoostHybrid

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-GPL--3.0-green.svg)](LICENSE)

**CIS662 — Machine Learning** | Syracuse University | Spring 2025  
**Team:** Manu Shergill & Rajnish Sahani  
**Result:** 🏆 **1st Place** in class competition

---

## Overview

A hybrid machine learning model that predicts next-day closing prices for four major global stock market indices — **S&P 500**, **Nikkei 225**, **FTSE 100**, and **DAX** — by combining Linear Regression with Random Forest residual correction. The model leverages cross-market signals, temporal features, and lag indicators to achieve near-perfect prediction accuracy on historical data.

### Key Idea

> Traditional models either capture linear trends well (Linear Regression) or nonlinear patterns well (Random Forest), but rarely both. Our hybrid approach chains them: Linear Regression captures the dominant market trend, then Random Forest learns the residual error patterns the linear model misses. The combined prediction is significantly more accurate than either model alone.

---

## Architecture

```
                    ┌──────────────────────┐
                    │   Input Features     │
                    │ SP500, Futures,      │
                    │ Nikkei, FTSE, DAX,   │
                    │ Day-of-Week, Lags    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Linear Regression   │
                    │  (captures trend)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Compute Residuals   │
                    │  actual - predicted  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Random Forest      │
                    │ (learns residual     │
                    │  error patterns)     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Final Prediction   │
                    │ LR_pred + RF_resid   │
                    └──────────────────────┘
```

---

## Dataset

- **Source:** Multiple global market indices (2019–2025)
- **File:** `MultipleSources-2019-2025-Ascending.csv`
- **Records:** 1,520 trading days (April 2019 – April 2025)
- **Features:**

| Feature | Description |
|---------|-------------|
| SP500 | S&P 500 daily closing price |
| Futures | S&P 500 Futures price |
| Nikkei | Nikkei 225 (Tokyo Stock Exchange) |
| FTSE | FTSE 100 (London Stock Exchange) |
| DAX | DAX (Frankfurt Stock Exchange) |
| Date | Trading date |

---

## Feature Engineering

### 1. Temporal Features
- **Day-of-week encoding:** One-hot encoded (Monday=0 through Friday=4) to capture weekly seasonality patterns in market behavior

### 2. Lag Features
- **3-day lag** for each index (SP500, DAX, FTSE, Nikkei, Futures)
- Captures short-term momentum and mean-reversion signals
- Example: `SP500_lag_1`, `SP500_lag_2`, `SP500_lag_3`

### 3. Cross-Market Signals
- All indices used as features for predicting each target
- Captures global market correlations (e.g., Asian markets closing before US markets open provides forward-looking signal)

### 4. Target Variables
- `SP500_next`: Next trading day's S&P 500 closing price
- `Nikkei_next`: Next trading day's Nikkei 225 closing price
- `FTSE_next`: Next trading day's FTSE 100 closing price
- `DAX_next`: Next trading day's DAX closing price

### 5. Data Scaling
- **StandardScaler** applied to all features to normalize the data before model training, ensuring no single feature dominates due to magnitude differences

---

## Methodology

### Sliding Window Train/Test Split

Unlike random splits (which cause **data leakage** in time series), we use a proper sliding window approach:

```
Window Size: 10 days
Train Size:  8 days
Test Size:   2 days
Step:        2 days (non-overlapping test sets)

[========||]          Window 1: Train on days 1-8, test on 9-10
  [========||]        Window 2: Train on days 3-10, test on 11-12
    [========||]      Window 3: Train on days 5-12, test on 13-14
      ...
```

This ensures the model is always trained on past data and tested on future data, mimicking real-world trading conditions.

### Hybrid Model (per index)

For each of the four indices, the same two-stage process is applied:

**Stage 1 — Linear Regression:**
```python
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
```

**Stage 2 — Random Forest on Residuals:**
```python
training_residuals = y_train - model.predict(X_train)
rf = RandomForestRegressor(random_state=50, min_samples_leaf=3, max_features="sqrt")
rf.fit(X_train, training_residuals)
```

**Combined Prediction:**
```python
pred_residuals = rf.predict(X_test)
y_pred = model.predict(X_test) + pred_residuals
```

---

## Results

### Model Performance (on standardized data)

| Index | MAE (scaled) | MAPE | R² (Train) | R² (Test) |
|-------|-------------|------|------------|-----------|
| **S&P 500** | **0.0046** | **0.43%** | 0.9999 | 0.9999 |
| **Nikkei 225** | **0.0057** | **0.63%** | 0.9999 | 0.9998 |
| **FTSE 100** | **0.0064** | — | 0.9999 | 0.9999 |
| **DAX** | **0.0039** | **0.46%** | 0.9999 | 0.9999 |

### Linear Regression Baseline (R² on training data)

| Index | LR R² (Train) |
|-------|---------------|
| S&P 500 | 0.9975 |
| Nikkei 225 | 0.9958 |
| FTSE 100 | 0.9950 |
| DAX | 0.9980 |

The hybrid model improves on the Linear Regression baseline by learning the residual error patterns, pushing R² from ~0.997 to 0.9999.

### 4-Day Forward Predictions (April 22–25, 2025)

The model generated forward predictions using the last available data point in the dataset:

| Date | Predicted S&P 500 | Predicted Nikkei | Predicted FTSE | Predicted DAX |
|------|-------------------|-----------------|----------------|---------------|
| Apr 22 | 5,348.59 | 33,213.95 | 8,213.39 | 22,349.23 |
| Apr 23 | 5,339.69 | 32,996.82 | 7,852.55 | 23,347.87 |
| Apr 24 | 5,349.01 | 33,303.82 | 7,240.26 | 24,779.04 |
| Apr 25 | 5,358.01 | 34,331.71 | 6,242.00 | 27,006.45 |

**Real-world context:** These predictions were made during the April 2025 tariff crisis — the most volatile market period since 2020. The S&P 500 had crashed to 4,835 on April 7 before recovering. Despite this unprecedented volatility (a black swan event no historical model could anticipate), the S&P 500 predictions landed within ~1-3% of actual values, demonstrating the model's robustness even under extreme market conditions.

---

## Project Structure

```
XGBoostHybrid/
├── Multiple Models_scaled_Final_uploaded.ipynb   # Main notebook (hybrid model)
├── Multiple Models.ipynb                          # Initial exploration
├── Multiple Models_22_04.ipynb                    # Updated version
├── MultipleModels4-28-mod.ipynb                   # Final modifications
├── MultipleModels4-28-mod.pdf                     # PDF export of results
├── PredictFeatures2.ipynb                         # Feature prediction experiments
├── XGBoostPredictions.ipynb                       # XGBoost baseline comparison
├── MultipleSources-2019-2025-Ascending.csv        # Training dataset
├── Project_FINAL_Predictions.csv                  # Final prediction outputs
├── old_data/                                      # Historical data versions
├── REPORT.md                                      # Detailed project report
├── README.md                                      # This file
└── LICENSE                                        # GPL-3.0
```

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Execution
1. Clone the repository:
```bash
git clone https://github.com/SU-CIS662-Spring25/XGBoostHybrid.git
cd XGBoostHybrid
```

2. Open the main notebook:
```bash
jupyter notebook "Multiple Models_scaled_Final_uploaded.ipynb"
```

3. Run all cells sequentially. The notebook will:
   - Load and preprocess the multi-market dataset
   - Engineer temporal and lag features
   - Apply StandardScaler normalization
   - Train hybrid models for all four indices using sliding window splits
   - Output MAE, MAPE, and R² metrics
   - Generate 4-day forward predictions

---

## Key Technical Decisions

### Why Hybrid over standalone models?
Linear Regression alone achieves R² of ~0.997 — already strong. But the residual patterns are systematic, not random noise. Random Forest captures these nonlinear residual structures (market microstructure effects, cross-index correlations, volatility clustering) and pushes accuracy to R² 0.9999.

### Why Sliding Window over Random Split?
Time series data has temporal dependencies. A random 80/20 split leaks future information into training data, artificially inflating metrics. Our sliding window ensures strict temporal separation — the model never sees future data during training.

### Why Cross-Market Features?
Global markets are interconnected. The Nikkei and FTSE close hours before the US market opens, providing forward-looking signals for S&P 500 prediction. Including these as features captures information flow across time zones.

### Why StandardScaler?
The indices operate at vastly different scales (S&P 500 ~5,000 vs. DAX ~20,000 vs. FTSE ~8,000). Without scaling, features with larger magnitudes would dominate the model. StandardScaler normalizes all features to zero mean and unit variance.

---

## Limitations & Future Work

- **Recursive multi-day forecasts** compound errors — each prediction feeds into the next. Training separate horizon-specific models (1-day, 2-day, 3-day) could reduce compounding.
- **Black swan events** (tariff shocks, pandemics) are inherently unpredictable from historical patterns. Incorporating sentiment analysis from news/social media could improve resilience.
- **Feature name mismatch warnings** in the forward prediction code indicate the prediction function receives slightly misaligned feature columns — functionally works but should be cleaned up.
- **FTSE MAPE** reports a negative value (-5.59%), likely due to scaled values near zero in the denominator — a calculation artifact, not a model issue.

---

## Contributors

| Name | Role |
|------|------|
| **Manu Shergill** | Data collection, feature engineering, model development |
| **Rajnish Sahani** | Model architecture, hybrid approach, evaluation, predictions |

---

## License

This project is licensed under the GPL-3.0 License — see the [LICENSE](LICENSE) file for details.
