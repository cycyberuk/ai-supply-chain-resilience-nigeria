"""
Demand Sensing Module — Module 1
AI-Driven Supply Chain Resilience in African E-Commerce
Dataset: Electric Sheep Africa — Nigeria Point of Sale Records (Parquet, manual download)
Output: Table 3 — MAPE / RMSE / R2 for seasonal naive baseline vs XGBoost
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA (manual Parquet file — no Hugging Face cache)
# ============================================================
PARQUET_FILE = "nigerian_retail_and_ecommerce_point_of_sale_records.parquet"

df = pd.read_parquet(PARQUET_FILE)
print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ============================================================
# 2. BUILD DAILY DEMAND SERIES
# ============================================================
# Ensure transaction_date is datetime
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# Aggregate by day: sum of items_count = unit demand
daily = (
    df.groupby(df["transaction_date"].dt.date)
      .agg(demand=("items_count", "sum"))
      .reset_index()
      .rename(columns={"transaction_date": "ds", "demand": "y"})
)
daily["ds"] = pd.to_datetime(daily["ds"])
daily = daily.sort_values("ds").reset_index(drop=True)

print("\nDaily series length:", len(daily))
print("Date range:", daily["ds"].min().date(), "to", daily["ds"].max().date())
print(daily.head())

# ============================================================
# 3. CHRONOLOGICAL TRAIN-TEST SPLIT (80/20)
# ============================================================
split_idx = int(len(daily) * 0.8)
train = daily.iloc[:split_idx].copy()
test  = daily.iloc[split_idx:].copy()

print(f"\nTrain: {train['ds'].min().date()} to {train['ds'].max().date()} ({len(train)} days)")
print(f"Test:  {test['ds'].min().date()} to {test['ds'].max().date()} ({len(test)} days)")

# ============================================================
# 4. SEASONAL NAIVE BASELINE (lag-7)
# ============================================================
full_series = pd.concat([train, test]).reset_index(drop=True).set_index("ds")
full_series["naive_pred"] = full_series["y"].shift(7)

test_naive = test.merge(
    full_series[["naive_pred"]].reset_index(),
    left_on="ds", right_on="ds", how="left"
).dropna(subset=["naive_pred"])

print("\nNaive predictions ready:", len(test_naive))

# ============================================================
# 5. FEATURE ENGINEERING FOR XGBOOST
# ============================================================
def create_features(df_series):
    df_feat = df_series.copy()
    df_feat["dayofweek"] = df_feat["ds"].dt.dayofweek
    df_feat["dayofmonth"] = df_feat["ds"].dt.day
    df_feat["month"] = df_feat["ds"].dt.month
    df_feat["is_weekend"] = (df_feat["dayofweek"] >= 5).astype(int)

    # Lag features — only past information
    for lag in [7, 14, 21, 28]:
        df_feat[f"lag_{lag}"] = df_feat["y"].shift(lag)

    # Rolling means — shift(1) prevents leakage
    for window in [7, 14, 28]:
        df_feat[f"roll_mean_{window}"] = df_feat["y"].shift(1).rolling(window).mean()

    return df_feat

daily_feat = create_features(daily)
daily_feat = daily_feat.dropna().reset_index(drop=True)

train_feat = daily_feat[daily_feat["ds"] < test["ds"].min()].copy()
test_feat  = daily_feat[daily_feat["ds"] >= test["ds"].min()].copy()

feature_cols = [c for c in daily_feat.columns if c not in ["ds", "y"]]
print("\nFeatures:", feature_cols)
print(f"Train feat: {len(train_feat)}, Test feat: {len(test_feat)}")

# ============================================================
# 6. TRAIN XGBOOST
# ============================================================
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)
model.fit(train_feat[feature_cols], train_feat["y"])
test_feat["xgb_pred"] = model.predict(test_feat[feature_cols])

# ============================================================
# 7. EVALUATION — MAPE, RMSE, R2
# ============================================================

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

def mase(y_true, y_pred, y_train, seasonality=7):
    """
    Mean Absolute Scaled Error.
    Scale = mean absolute error of the seasonal naive forecast on the training set.
    MASE < 1 means the model beats the seasonal naive forecast.
    """
    y_train = np.asarray(y_train)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = naive_errors.mean()
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale

def evaluate(y_true, y_pred, y_train, label):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    m    = mase(y_true, y_pred, y_train, seasonality=7)
    print(f"{label:25s} | MAPE: {mape:.4f} | RMSE: {rmse:.2f} | MASE: {m:.4f} | R2: {r2:.4f}")
    return {"Model": label, "MAPE": mape, "RMSE": rmse, "MASE": m, "R2": r2}

results = []

# Seasonal naive baseline
results.append(evaluate(
    test_naive["y"].values,
    test_naive["naive_pred"].values,
    train["y"].values,
    "Seasonal naive (lag-7)"
))

# XGBoost
results.append(evaluate(
    test_feat["y"].values,
    test_feat["xgb_pred"].values,
    train["y"].values,
    "XGBoost"
))

results_df = pd.DataFrame(results)
print("\n=== Table 3 (actual numbers) ===")
print(results_df.to_string(index=False))

# Relative improvement of XGBoost over baseline
naive_row = results_df[results_df["Model"].str.contains("naive")].iloc[0]
xgb_row   = results_df[results_df["Model"] == "XGBoost"].iloc[0]

print("\n=== Relative improvement (XGBoost vs Seasonal Naive) ===")
print(f"MAPE reduction: {((naive_row['MAPE'] - xgb_row['MAPE']) / naive_row['MAPE']) * 100:.1f}%")
print(f"RMSE reduction: {((naive_row['RMSE'] - xgb_row['RMSE']) / naive_row['RMSE']) * 100:.1f}%")

# Save results for paper / later modules
results_df.to_csv("table3_demand_sensing_results.csv", index=False)
print("\nSaved: table3_demand_sensing_results.csv")

# Save XGBoost forecasts for Module 2 (inventory)
test_feat[["ds", "y", "xgb_pred"]].to_csv("module1_demand_forecasts.csv", index=False)
print("Saved: module1_demand_forecasts.csv (for Module 2 inventory)")