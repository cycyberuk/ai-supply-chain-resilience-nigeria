"""
Last-Mile ETA Prediction and Risk Classification — Module 3A
AI-Driven Supply Chain Resilience in African E-Commerce
Input:  nigerian_retail_and_ecommerce_last_mile_delivery_data.parquet
Output: Table 5A — ETA regression metrics + risk classification metrics
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 1. LOAD DATA
# ============================================================
PARQUET = "nigerian_retail_and_ecommerce_last_mile_delivery_data.parquet"
df = pd.read_parquet(PARQUET)
print("Raw shape:", df.shape)
print("Status distribution:\n", df["delivery_status"].value_counts())

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
# Parse pickup_time
df["pickup_time"] = pd.to_datetime(df["pickup_time"])
df["hour"] = df["pickup_time"].dt.hour
df["dayofweek"] = df["pickup_time"].dt.dayofweek
df["month"] = df["pickup_time"].dt.month

# Categorical encoding
df["logistics_enc"] = LabelEncoder().fit_transform(df["logistics_company"])
df["pickup_city_enc"] = LabelEncoder().fit_transform(df["pickup_city"])
df["delivery_city_enc"] = LabelEncoder().fit_transform(df["delivery_city"])

# Same-city flag (pickup and delivery in same city)
df["same_city"] = (df["pickup_city"] == df["delivery_city"]).astype(int)

# Feature set
feature_cols = [
    "distance_km", "hour", "dayofweek", "month",
    "logistics_enc", "pickup_city_enc", "delivery_city_enc", "same_city"
]

# ============================================================
# 3. REGRESSION — PREDICT DELIVERY DURATION (ETA)
# ============================================================
print("\n=== ETA Regression ===")
X = df[feature_cols].values
y = df["delivery_duration_minutes"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_leaf=10,
    random_state=42, n_jobs=-1
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# Baseline: predict the mean
baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
baseline_mae  = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print(f"Random Forest Regressor")
print(f"  MAE:  {mae:.2f} minutes")
print(f"  RMSE: {rmse:.2f} minutes")
print(f"  R2:   {r2:.4f}")
print(f"Baseline (predict mean)")
print(f"  MAE:  {baseline_mae:.2f} minutes")
print(f"  RMSE: {baseline_rmse:.2f} minutes")
print(f"Improvement: MAE {(1 - mae/baseline_mae)*100:.1f}%, "
      f"RMSE {(1 - rmse/baseline_rmse)*100:.1f}%")

# ============================================================
# 4. CLASSIFICATION — PREDICT DELIVERY FAILURE / RETURN
# ============================================================
print("\n=== Risk Classification (delivered vs failed/returned) ===")
# Binary: 1 = not delivered (failed or returned), 0 = delivered
df["risk"] = (df["delivery_status"] != "delivered").astype(int)

Xc = df[feature_cols].values
yc = df["risk"].values

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc
)

clf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_leaf=10,
    random_state=42, n_jobs=-1, class_weight="balanced"
)
clf.fit(Xc_train, yc_train)

yc_pred  = clf.predict(Xc_test)
yc_proba = clf.predict_proba(Xc_test)[:, 1]

acc = accuracy_score(yc_test, yc_pred)
auc = roc_auc_score(yc_test, yc_proba)

print(f"Random Forest Classifier")
print(f"  Accuracy: {acc:.4f}")
print(f"  AUC:      {auc:.4f}")
print("\nClassification report:")
print(classification_report(yc_test, yc_pred,
                            target_names=["delivered", "failed/returned"]))

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
print("\n=== Feature Importance (regression) ===")
importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": reg.feature_importances_
}).sort_values("Importance", ascending=False)
print(importance.to_string(index=False))

# ============================================================
# 6. SAVE TABLE 5A
# ============================================================
table5a = pd.DataFrame([
    {"Model": "Baseline (mean)", "MAE": round(baseline_mae, 2),
     "RMSE": round(baseline_rmse, 2), "R2": "—"},
    {"Model": "Random Forest", "MAE": round(mae, 2),
     "RMSE": round(rmse, 2), "R2": round(r2, 4)},
])
table5a.to_csv("table5a_eta_regression.csv", index=False)
print("\nSaved: table5a_eta_regression.csv")

table5a_cls = pd.DataFrame([
    {"Model": "Random Forest", "Accuracy": round(acc, 4),
     "AUC": round(auc, 4), "Positive class": "failed/returned"}
])
table5a_cls.to_csv("table5a_risk_classification.csv", index=False)
print("Saved: table5a_risk_classification.csv")

# ============================================================
# 7. PLOT: PREDICTED VS ACTUAL (for paper figure)
# ============================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:2000], y_pred[:2000], alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.xlabel("Actual delivery duration (minutes)")
plt.ylabel("Predicted delivery duration (minutes)")
plt.title("ETA Prediction: Predicted vs Actual (n=2000 sample)")
plt.tight_layout()
plt.savefig("fig_eta_prediction.png", dpi=300)
print("Saved: fig_eta_prediction.png")