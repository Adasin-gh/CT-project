# ============================================================
# TASK 1: BIG DATA ANALYSIS USING DASK
# ============================================================
# Tool Used  : Dask (scalable alternative to Pandas)
# Dataset    : Synthetic Sales Dataset (1 Million rows)
# Deliverable: Script with insights from big data processing
# ============================================================

# ── STEP 1: Install required libraries ──────────────────────
# Run this in terminal before running the script:
# pip install dask pandas numpy matplotlib pyarrow

import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("        TASK 1: BIG DATA ANALYSIS USING DASK")
print("=" * 60)

# ── STEP 2: Generate a Large Synthetic Dataset ───────────────
print("\n[1/6] Generating large synthetic dataset (1 Million rows)...")

np.random.seed(42)
n_rows = 1_000_000  # 1 Million rows to demonstrate big data

data = {
    'order_id':     np.arange(1, n_rows + 1),
    'date':         pd.date_range(start='2020-01-01', periods=n_rows, freq='1min'),
    'product':      np.random.choice(['Laptop', 'Phone', 'Tablet', 'Watch', 'Earbuds'], n_rows),
    'category':     np.random.choice(['Electronics', 'Accessories', 'Wearables'], n_rows),
    'region':       np.random.choice(['North', 'South', 'East', 'West'], n_rows),
    'sales_amount': np.random.uniform(500, 50000, n_rows).round(2),
    'quantity':     np.random.randint(1, 10, n_rows),
    'discount':     np.random.uniform(0, 0.3, n_rows).round(2),
    'customer_age': np.random.randint(18, 70, n_rows),
    'rating':       np.random.choice([1, 2, 3, 4, 5], n_rows),
}

# Save as CSV to simulate a real big data file
df_pandas = pd.DataFrame(data)
df_pandas.to_csv('sales_data_large.csv', index=False)
print(f"   ✓ Dataset created: {n_rows:,} rows × {len(data)} columns")
print(f"   ✓ Saved as 'sales_data_large.csv'")

# ── STEP 3: Load Data with Dask ───────────────────────────────
print("\n[2/6] Loading data with DASK (scalable loading)...")

start_time = time.time()
ddf = dd.read_csv('sales_data_large.csv', parse_dates=['date'])
load_time = time.time() - start_time

print(f"   ✓ Dask DataFrame loaded in {load_time:.3f} seconds")
print(f"   ✓ Number of partitions: {ddf.npartitions}")
print(f"   ✓ Columns: {list(ddf.columns)}")

# ── STEP 4: Basic Data Exploration ───────────────────────────
print("\n[3/6] Basic Data Exploration...")

total_rows = len(ddf)
print(f"   ✓ Total rows: {total_rows:,}")
print(f"   ✓ Data types:\n{ddf.dtypes}")

print("\n   Computing summary statistics (Dask lazy computation)...")
stats = ddf[['sales_amount', 'quantity', 'discount', 'rating']].describe().compute()
print(stats.to_string())

# ── STEP 5: Data Analysis & Aggregations ─────────────────────
print("\n[4/6] Running Big Data Aggregations with Dask...")

# --- Analysis 1: Total Sales by Region ---
print("\n   📊 Analysis 1: Total Sales by Region")
sales_by_region = (
    ddf.groupby('region')['sales_amount']
    .sum()
    .compute()
    .sort_values(ascending=False)
)
print(sales_by_region.to_string())

# --- Analysis 2: Average Sales by Product ---
print("\n   📊 Analysis 2: Average Sales by Product")
avg_by_product = (
    ddf.groupby('product')['sales_amount']
    .mean()
    .compute()
    .sort_values(ascending=False)
)
print(avg_by_product.round(2).to_string())

# --- Analysis 3: Total Revenue by Category ---
print("\n   📊 Analysis 3: Total Revenue by Category")
revenue_by_category = (
    ddf.groupby('category')['sales_amount']
    .sum()
    .compute()
    .sort_values(ascending=False)
)
print(revenue_by_category.to_string())

# --- Analysis 4: Monthly Sales Trend ---
print("\n   📊 Analysis 4: Monthly Sales Trend (first 12 months)")
ddf['month'] = ddf['date'].dt.to_period('M').astype(str)
monthly_sales = (
    ddf.groupby('month')['sales_amount']
    .sum()
    .compute()
    .sort_index()
    .head(12)
)
print(monthly_sales.to_string())

# --- Analysis 5: Average Rating by Product ---
print("\n   📊 Analysis 5: Average Customer Rating by Product")
rating_by_product = (
    ddf.groupby('product')['rating']
    .mean()
    .compute()
    .sort_values(ascending=False)
)
print(rating_by_product.round(2).to_string())

# --- Analysis 6: High Value Orders (Sales > 40,000) ---
print("\n   📊 Analysis 6: High Value Orders (Sales > ₹40,000)")
high_value = ddf[ddf['sales_amount'] > 40000]
high_value_count = len(high_value.compute())
print(f"   Total high-value orders: {high_value_count:,}")

# ── STEP 6: Visualizations ────────────────────────────────────
print("\n[5/6] Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Task 1: Big Data Analysis Insights\n(Dataset: 1 Million Sales Records)',
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1: Sales by Region
axes[0, 0].bar(sales_by_region.index, sales_by_region.values / 1e6,
               color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'])
axes[0, 0].set_title('Total Sales by Region (in Millions)')
axes[0, 0].set_xlabel('Region')
axes[0, 0].set_ylabel('Sales (₹ Millions)')
axes[0, 0].tick_params(axis='x', rotation=0)

# Plot 2: Avg Sales by Product
axes[0, 1].barh(avg_by_product.index, avg_by_product.values,
                color=['#9B59B6', '#1ABC9C', '#E67E22', '#E74C3C', '#3498DB'])
axes[0, 1].set_title('Average Sales Amount by Product')
axes[0, 1].set_xlabel('Average Sales (₹)')

# Plot 3: Revenue by Category - Pie chart
axes[0, 2].pie(revenue_by_category.values, labels=revenue_by_category.index,
               autopct='%1.1f%%', colors=['#E74C3C', '#3498DB', '#2ECC71'],
               startangle=90)
axes[0, 2].set_title('Revenue Distribution by Category')

# Plot 4: Monthly Sales Trend
axes[1, 0].plot(range(len(monthly_sales)), monthly_sales.values / 1e6,
                marker='o', color='#E74C3C', linewidth=2, markersize=5)
axes[1, 0].set_title('Monthly Sales Trend (First 12 Months)')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Sales (₹ Millions)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 5: Rating by Product
axes[1, 1].bar(rating_by_product.index, rating_by_product.values,
               color=['#F1C40F', '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'])
axes[1, 1].set_title('Average Customer Rating by Product')
axes[1, 1].set_xlabel('Product')
axes[1, 1].set_ylabel('Avg Rating (out of 5)')
axes[1, 1].set_ylim(0, 5)

# Plot 6: Sales Distribution Histogram
sample_sales = ddf['sales_amount'].sample(frac=0.01).compute()
axes[1, 2].hist(sample_sales, bins=50, color='#3498DB', edgecolor='white', alpha=0.8)
axes[1, 2].set_title('Sales Amount Distribution (1% Sample)')
axes[1, 2].set_xlabel('Sales Amount (₹)')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('task1_big_data_insights.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✓ Chart saved as 'task1_big_data_insights.png'")

# ── STEP 7: Final Insights Summary ───────────────────────────
print("\n[6/6] KEY INSIGHTS SUMMARY")
print("=" * 60)
print(f"✅ Dataset Size         : {total_rows:,} rows (Big Data scale)")
print(f"✅ Top Region by Sales  : {sales_by_region.idxmax()}")
print(f"✅ Top Product (Avg)    : {avg_by_product.idxmax()}")
print(f"✅ Best Rated Product   : {rating_by_product.idxmax()}")
print(f"✅ High Value Orders    : {high_value_count:,} orders above ₹40,000")
print(f"✅ Total Revenue        : ₹{ddf['sales_amount'].sum().compute():,.0f}")
print("=" * 60)
print("\n✅ TASK 1 COMPLETE!")