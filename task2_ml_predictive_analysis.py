# ============================================================
# TASK 2: PREDICTIVE ANALYSIS USING MACHINE LEARNING
# ============================================================
# Model Used : Random Forest Regressor + Linear Regression
# Dataset    : Synthetic Sales Dataset
# Deliverable: Feature Selection, Model Training & Evaluation
# ============================================================

# ── STEP 0: Install required libraries ──────────────────────
# pip install scikit-learn pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)

print("=" * 60)
print("    TASK 2: PREDICTIVE ANALYSIS USING MACHINE LEARNING")
print("=" * 60)

# ── STEP 1: Generate / Load Dataset ──────────────────────────
print("\n[1/7] Creating synthetic sales dataset...")

np.random.seed(42)
n = 50_000   # 50k rows

products   = ['Laptop', 'Phone', 'Tablet', 'Watch', 'Earbuds']
categories = ['Electronics', 'Accessories', 'Wearables']
regions    = ['North', 'South', 'East', 'West']
product_base = {'Laptop':40000,'Phone':25000,'Tablet':20000,
                'Watch':15000,'Earbuds':5000}

rows = []
for i in range(n):
    prod   = np.random.choice(products)
    cat    = np.random.choice(categories)
    reg    = np.random.choice(regions)
    qty    = np.random.randint(1, 10)
    disc   = round(np.random.uniform(0, 0.3), 2)
    age    = np.random.randint(18, 70)
    rating = np.random.choice([1, 2, 3, 4, 5])
    base   = product_base[prod]
    sales  = round(base * qty * (1 - disc) + np.random.normal(0, 500), 2)
    rows.append([prod, cat, reg, qty, disc, age, rating, sales])

df = pd.DataFrame(rows, columns=[
    'product','category','region',
    'quantity','discount','customer_age','rating','sales_amount'])

print(f"   ✓ Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(df.head(3).to_string())

# ── STEP 2: EDA ───────────────────────────────────────────────
print("\n[2/7] Exploratory Data Analysis...")
print(df.describe().round(2).to_string())
print(f"\n   Missing values:\n{df.isnull().sum().to_string()}")

# ── STEP 3: Feature Engineering ──────────────────────────────
print("\n[3/7] Feature Engineering & Encoding...")

df_ml = df.copy()

# Encode categorical columns
le = LabelEncoder()
for col in ['product', 'category', 'region']:
    df_ml[col + '_enc'] = le.fit_transform(df_ml[col])
    print(f"   ✓ Encoded '{col}'")

# Feature columns & target
FEATURES = ['product_enc', 'category_enc', 'region_enc',
            'quantity', 'discount', 'customer_age', 'rating']
TARGET   = 'sales_amount'

X = df_ml[FEATURES]
y = df_ml[TARGET]

print(f"\n   Features used : {FEATURES}")
print(f"   Target column : {TARGET}")

# ── STEP 4: Feature Correlation ──────────────────────────────
print("\n[4/7] Checking feature correlation with target...")
corr = df_ml[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
print(corr.round(4).to_string())

# ── STEP 5: Train-Test Split & Scaling ───────────────────────
print("\n[5/7] Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   ✓ Train set: {X_train.shape[0]:,} rows")
print(f"   ✓ Test set : {X_test.shape[0]:,} rows")

# ── STEP 6: Model Training & Evaluation ──────────────────────
print("\n[6/7] Training Models...")

# --- Model 1: Linear Regression ---
print("\n   🔵 Model 1: Linear Regression")
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print(f"   MAE  : ₹{mae_lr:,.2f}")
print(f"   RMSE : ₹{rmse_lr:,.2f}")
print(f"   R²   : {r2_lr:.4f}  ({r2_lr*100:.1f}% variance explained)")

# --- Model 2: Random Forest Regressor ---
print("\n   🟢 Model 2: Random Forest Regressor")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

print(f"   MAE  : ₹{mae_rf:,.2f}")
print(f"   RMSE : ₹{rmse_rf:,.2f}")
print(f"   R²   : {r2_rf:.4f}  ({r2_rf*100:.1f}% variance explained)")

# Feature importance from RF
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n   📌 Feature Importances (Random Forest):")
print(feat_imp.round(4).to_string())

# ── STEP 7: Visualizations ────────────────────────────────────
print("\n[7/7] Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Task 2: ML Predictive Analysis\n(Target: Sales Amount Prediction)',
             fontsize=14, fontweight='bold')

# Plot 1: Correlation heatmap
corr_mat = df_ml[FEATURES + [TARGET]].corr()
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[0, 0], linewidths=0.5)
axes[0, 0].set_title('Feature Correlation Heatmap')

# Plot 2: Feature Importance
axes[0, 1].barh(feat_imp.index, feat_imp.values,
                color=['#2ECC71' if v > 0.1 else '#3498DB' for v in feat_imp.values])
axes[0, 1].set_title('Feature Importance (Random Forest)')
axes[0, 1].set_xlabel('Importance Score')

# Plot 3: Model Comparison
models = ['Linear Regression', 'Random Forest']
r2vals = [r2_lr, r2_rf]
colors = ['#E74C3C', '#2ECC71']
bars = axes[0, 2].bar(models, r2vals, color=colors)
axes[0, 2].set_title('Model Comparison (R² Score)')
axes[0, 2].set_ylabel('R² Score')
axes[0, 2].set_ylim(0, 1.1)
for bar, val in zip(bars, r2vals):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontweight='bold')

# Plot 4: Actual vs Predicted - Linear Regression
sample = np.random.choice(len(y_test), 500, replace=False)
axes[1, 0].scatter(y_test.iloc[sample], y_pred_lr[sample],
                   alpha=0.4, color='#E74C3C', s=10)
axes[1, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'k--', linewidth=1)
axes[1, 0].set_title(f'Linear Regression\nActual vs Predicted (R²={r2_lr:.3f})')
axes[1, 0].set_xlabel('Actual Sales (₹)')
axes[1, 0].set_ylabel('Predicted Sales (₹)')

# Plot 5: Actual vs Predicted - Random Forest
axes[1, 1].scatter(y_test.iloc[sample], y_pred_rf[sample],
                   alpha=0.4, color='#2ECC71', s=10)
axes[1, 1].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'k--', linewidth=1)
axes[1, 1].set_title(f'Random Forest\nActual vs Predicted (R²={r2_rf:.3f})')
axes[1, 1].set_xlabel('Actual Sales (₹)')
axes[1, 1].set_ylabel('Predicted Sales (₹)')

# Plot 6: Residuals distribution
residuals_rf = y_test.values - y_pred_rf
axes[1, 2].hist(residuals_rf, bins=60, color='#9B59B6',
                edgecolor='white', alpha=0.8)
axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[1, 2].set_title('Residuals Distribution (Random Forest)')
axes[1, 2].set_xlabel('Residual (Actual - Predicted)')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('task2_ml_insights.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✓ Chart saved as 'task2_ml_insights.png'")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("         FINAL MODEL EVALUATION SUMMARY")
print("=" * 60)
print(f"{'Metric':<20} {'Linear Reg':>15} {'Random Forest':>15}")
print("-" * 52)
print(f"{'MAE (₹)':<20} {mae_lr:>15,.2f} {mae_rf:>15,.2f}")
print(f"{'RMSE (₹)':<20} {rmse_lr:>15,.2f} {rmse_rf:>15,.2f}")
print(f"{'R² Score':<20} {r2_lr:>15.4f} {r2_rf:>15.4f}")
print("=" * 60)
winner = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
print(f"\n🏆 Best Model   : {winner}")
print(f"   Top Feature  : {feat_imp.idxmax()} (importance: {feat_imp.max():.4f})")
print("\n✅ TASK 2 COMPLETE!")