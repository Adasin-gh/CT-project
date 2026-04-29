# 📊 Data Analysis Projects

> A complete collection of 4 data analysis projects covering Big Data Processing, Machine Learning, Interactive Dashboard Development, and Natural Language Processing.

---

## 🗂️ Projects Overview

| # | Task | Topic | Tools Used |
|---|------|-------|-----------|
| 1 | Big Data Analysis | Dask + 1M row dataset | Python, Dask, Pandas, Matplotlib |
| 2 | Predictive ML Model | Sales Amount Regression | Scikit-learn, Seaborn |
| 3 | Interactive Dashboard | Sales Analytics UI | HTML, CSS, JavaScript, Chart.js |
| 4 | Sentiment Analysis | NLP on Product Reviews | TF-IDF, Logistic Regression, NB |

---

## 📁 Repository Structure

```
CT-project/
│
├── 📄 task1_big_data_analysis.py       # Big Data Analysis using Dask
├── 📄 task2_ml_predictive_analysis.py  # ML Predictive Model
├── 📄 task3_dashboard.html             # Interactive Sales Dashboard
├── 📄 task4_sentiment_analysis.py      # NLP Sentiment Analysis
│
├── 📊 task1_big_data_insights.png      # Charts – Task 1
├── 📊 task2_ml_insights.png            # Charts – Task 2
├── 📊 task4_sentiment_insights.png     # Charts – Task 4
│
├── 🗃️ sales_data.csv                   # Sales Dataset (Task 1, 2, 3)
├── 🗃️ sales_data_large.csv             # Large Sales Dataset (1M rows)
├── 🗃️ reviews_data.csv                 # Product Reviews Dataset (Task 4)
│
└── 📝 README.md
```

---

## 🚀 Task 1 – Big Data Analysis using Dask

### Objective
Perform scalable analysis on a **1 Million row** synthetic sales dataset using **Dask** — a parallel computing library that handles datasets larger than memory.

### Key Features
- Generates 1,000,000 row dataset with 10 columns
- Demonstrates Dask's lazy evaluation and parallel processing
- 6 in-depth analyses: Regional Sales, Product Averages, Monthly Trends, Ratings, Revenue by Category, High-Value Orders

### Insights Found
- 📍 **Top Region by Sales:** East
- 🛍️ **Top Product (Avg Sales):** Watch
- ⭐ **Best Rated Product:** Tablet
- 💰 **High Value Orders (>₹40K):** 202,250 orders
- 💵 **Total Revenue:** ₹25,25,33,27,550

### How to Run
```bash
pip install dask pandas numpy matplotlib pyarrow
python3 task1_big_data_analysis.py
```

---

## 🤖 Task 2 – Predictive Analysis using Machine Learning

### Objective
Build and compare **regression models** to predict sales amount based on product features.

### Models Used
| Model | Performance |
|-------|-------------|
| Linear Regression | Baseline |
| **Random Forest** ✅ | **Best Model** |

### Key Features
- Label Encoding for categorical features
- StandardScaler for normalization
- Feature Importance visualization
- Actual vs Predicted charts
- Residuals distribution analysis

### Insights Found
- 🏆 **Best Model:** Random Forest Regressor
- 🔑 **Top Feature:** `quantity` (highest importance score)
- 📈 Random Forest explains significantly more variance than Linear Regression

### How to Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python3 task2_ml_predictive_analysis.py
```

---

## 📈 Task 3 – Interactive Dashboard Development

### Objective
Create a **fully functional interactive dashboard** to visualize the sales dataset with real-time filter controls.

### Features
- **5 KPI Cards** — Total Revenue, Orders, AOV, Avg Rating, Top Product
- **Bar Chart** — Sales by Region
- **Donut Chart** — Revenue by Product
- **Pie Chart** — Category Split
- **Line Chart** — Monthly Revenue Trend
- **Radar Chart** — Avg Rating per Product
- **Performance Table** — Regional breakdown with revenue share bars
- **Live Filters** — Filter by Region, Product, Category (all charts update instantly)

### How to Run
```
Simply open task3_dashboard.html in any web browser.
No installation required.
```

---

## 💬 Task 4 – Sentiment Analysis using NLP

### Objective
Perform **sentiment classification** on 2,000 product reviews using NLP techniques to identify positive, negative, and neutral sentiments.

### Pipeline
```
Raw Reviews → Text Cleaning → Stopword Removal → TF-IDF Vectorization → ML Classifier → Sentiment Label
```

### Models Used
| Model | Performance |
|-------|-------------|
| **Logistic Regression** ✅ | **Best Accuracy** |
| Multinomial Naive Bayes | Good Baseline |

### Key Features
- Text preprocessing: lowercasing, punctuation removal, stopword removal
- TF-IDF Vectorizer with unigrams + bigrams (500 features)
- Live prediction demo on new reviews
- Confusion matrix, keyword analysis, per-product sentiment breakdown

### Insights Found
- 😊 **~50%** of reviews are Positive
- 😞 **~30%** of reviews are Negative
- 😐 **~20%** of reviews are Neutral
- 🔑 Top positive words: *amazing, excellent, fantastic, superb*
- 🔑 Top negative words: *terrible, waste, horrible, useless*

### How to Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python3 task4_sentiment_analysis.py
```

---

## 📦 Datasets

### sales_data.csv
| Column | Type | Description |
|--------|------|-------------|
| order_id | int | Unique order ID |
| date | date | Order date |
| product | str | Product name |
| category | str | Product category |
| region | str | Sales region |
| quantity | int | Units ordered |
| discount | float | Discount (0–0.3) |
| customer_age | int | Customer age |
| rating | int | Rating (1–5) |
| sales_amount | float | Total sale value (₹) |

### reviews_data.csv
| Column | Type | Description |
|--------|------|-------------|
| review_id | int | Unique review ID |
| product | str | Product reviewed |
| review | str | Raw review text |
| sentiment | str | positive / negative / neutral |
| sentiment_label | int | 1 / -1 / 0 |
| rating | int | Star rating (1–5) |

---

## ⚙️ Installation

Install all dependencies at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn dask[dataframe] pyarrow
```

---

## 📌 Notes
- All datasets are **synthetically generated** for demonstration purposes
- Task 3 dashboard runs **entirely in the browser** — no backend needed
- All Python scripts include **detailed comments** for readability
- Charts are automatically saved as `.png` files in the project folder