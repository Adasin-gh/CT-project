# ============================================================
# TASK 4: SENTIMENT ANALYSIS USING NLP
# ============================================================
# Techniques : TF-IDF Vectorization + ML Classifiers
# Dataset    : Product Reviews (reviews_data.csv)
# Deliverable: Data Preprocessing, Model, Insights
# ============================================================

# ── Install required libraries ───────────────────────────────
# pip install scikit-learn pandas numpy matplotlib seaborn nltk wordcloud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from sklearn.pipeline import Pipeline

print("=" * 60)
print("       TASK 4: SENTIMENT ANALYSIS USING NLP")
print("=" * 60)

# ── STEP 1: Load Dataset ─────────────────────────────────────
print("\n[1/7] Loading reviews dataset...")

try:
    df = pd.read_csv('reviews_data.csv')
    print(f"   ✓ Loaded existing reviews_data.csv: {df.shape}")
except FileNotFoundError:
    print("   ℹ Generating synthetic reviews dataset...")
    np.random.seed(42)
    n = 2000
    positive_reviews = [
        'This product is absolutely amazing, love it!',
        'Excellent quality, very happy with my purchase.',
        'Works perfectly, exceeded my expectations.',
        'Great value for money, highly recommend.',
        'Outstanding performance, will buy again.',
        'Fantastic product, delivery was fast too.',
        'Best purchase I made this year, very satisfied.',
        'Superb quality and great customer service.',
        'Really impressed with the build quality.',
        'Wonderful experience, product is top notch.',
    ]
    negative_reviews = [
        'Terrible product, broke after one week.',
        'Very disappointed, does not work as advertised.',
        'Waste of money, poor quality.',
        'Horrible experience, never buying again.',
        'Product stopped working after a few days.',
        'Very bad quality, not worth the price.',
        'Completely useless, returned immediately.',
        'Worst purchase ever, avoid this product.',
        'Not satisfied at all, very poor performance.',
        'Defective product, customer service unhelpful.',
    ]
    neutral_reviews = [
        'Product is okay, nothing special.',
        'Average quality, does the job.',
        'Decent product, meets basic expectations.',
        'It is alright, neither good nor bad.',
        'Works fine, but nothing extraordinary.',
        'Satisfactory product, as described.',
        'Normal product, average performance.',
        'Okay for the price, not bad.',
        'Neither impressed nor disappointed.',
        'Product is acceptable, works as expected.',
    ]
    reviews, sentiments, labels = [], [], []
    for _ in range(n):
        s = np.random.choice(['positive','negative','neutral'], p=[0.5,0.3,0.2])
        if s == 'positive':
            reviews.append(np.random.choice(positive_reviews)); labels.append(1)
        elif s == 'negative':
            reviews.append(np.random.choice(negative_reviews)); labels.append(-1)
        else:
            reviews.append(np.random.choice(neutral_reviews)); labels.append(0)
        sentiments.append(s)
    products = np.random.choice(['Laptop','Phone','Tablet','Watch','Earbuds'], n)
    ratings  = [5 if s=='positive' else (1 if s=='negative' else 3) for s in sentiments]
    ratings  = [max(1,min(5, r + np.random.randint(-1,2))) for r in ratings]
    df = pd.DataFrame({'review_id': range(1,n+1), 'product': products,
                       'review': reviews, 'sentiment': sentiments,
                       'sentiment_label': labels, 'rating': ratings})
    df.to_csv('reviews_data.csv', index=False)

print(df.head(5).to_string())

# ── STEP 2: EDA ───────────────────────────────────────────────
print("\n[2/7] Exploratory Data Analysis...")
print(f"   Total reviews   : {len(df):,}")
print(f"   Missing values  : {df.isnull().sum().sum()}")
print(f"\n   Sentiment Distribution:")
print(df['sentiment'].value_counts().to_string())
print(f"\n   Reviews per Product:")
print(df['product'].value_counts().to_string())

# ── STEP 3: Text Preprocessing ───────────────────────────────
print("\n[3/7] Text Preprocessing...")

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

df['clean_review'] = df['review'].apply(preprocess_text)

# Common stopwords (manual list, no nltk needed)
STOPWORDS = {'the','a','an','is','it','in','on','at','to','for','of',
             'and','or','but','this','that','with','my','i','very',
             'was','be','are','as','not','no','so','do','its'}

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in STOPWORDS])

df['clean_review'] = df['clean_review'].apply(remove_stopwords)
print(f"   ✓ Text cleaned & preprocessed")
print(f"   Sample cleaned review: '{df['clean_review'].iloc[0]}'")

# ── STEP 4: Feature Extraction (TF-IDF) ──────────────────────
print("\n[4/7] Feature Extraction using TF-IDF...")

X = df['clean_review']
y = df['sentiment']  # positive / negative / neutral

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   ✓ Train size : {len(X_train):,}")
print(f"   ✓ Test size  : {len(X_test):,}")

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"   ✓ TF-IDF vocabulary size: {len(tfidf.vocabulary_):,}")
print(f"   ✓ Feature matrix shape  : {X_train_tfidf.shape}")

# ── STEP 5: Model Training ────────────────────────────────────
print("\n[5/7] Training Sentiment Models...")

# --- Model 1: Logistic Regression ---
print("\n   🔵 Model 1: Logistic Regression")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"   Accuracy: {acc_lr*100:.2f}%")
print(classification_report(y_test, y_pred_lr))

# --- Model 2: Naive Bayes ---
print("\n   🟢 Model 2: Multinomial Naive Bayes")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"   Accuracy: {acc_nb*100:.2f}%")
print(classification_report(y_test, y_pred_nb))

# ── STEP 6: Live Prediction Demo ─────────────────────────────
print("\n[6/7] Live Prediction Demo...")
demo_reviews = [
    "This laptop is absolutely fantastic, best purchase ever!",
    "Terrible quality, broke after two days, waste of money.",
    "The phone works okay, nothing special about it.",
    "Amazing earbuds, crystal clear sound, highly recommend!",
    "Very disappointed with the tablet, poor performance.",
]
print(f"\n   {'Review':<55} {'Predicted':>10}")
print("   " + "-"*67)
for rev in demo_reviews:
    cleaned = remove_stopwords(preprocess_text(rev))
    vec     = tfidf.transform([cleaned])
    pred    = lr.predict(vec)[0]
    emoji   = '😊' if pred=='positive' else ('😞' if pred=='negative' else '😐')
    print(f"   {rev[:52]:<55} {emoji} {pred:>10}")

# ── STEP 7: Visualizations ────────────────────────────────────
print("\n[7/7] Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Task 4: Sentiment Analysis – NLP Insights',
             fontsize=14, fontweight='bold')

COLORS = {'positive':'#2ECC71','negative':'#E74C3C','neutral':'#F39C12'}

# Plot 1: Sentiment Distribution
sent_counts = df['sentiment'].value_counts()
axes[0,0].bar(sent_counts.index,
              sent_counts.values,
              color=[COLORS[s] for s in sent_counts.index])
axes[0,0].set_title('Sentiment Distribution')
axes[0,0].set_xlabel('Sentiment')
axes[0,0].set_ylabel('Count')

# Plot 2: Sentiment per Product
pivot = df.groupby(['product','sentiment']).size().unstack(fill_value=0)
pivot.plot(kind='bar', ax=axes[0,1],
           color=[COLORS.get(c,'grey') for c in pivot.columns])
axes[0,1].set_title('Sentiment per Product')
axes[0,1].set_xlabel('Product')
axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=30)
axes[0,1].legend(title='Sentiment', fontsize=9)

# Plot 3: Confusion Matrix - Logistic Regression
labels_order = ['positive','negative','neutral']
cm = confusion_matrix(y_test, y_pred_lr, labels=labels_order)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_order, yticklabels=labels_order, ax=axes[0,2])
axes[0,2].set_title(f'Confusion Matrix – Logistic Reg\n(Accuracy: {acc_lr*100:.1f}%)')
axes[0,2].set_xlabel('Predicted')
axes[0,2].set_ylabel('Actual')

# Plot 4: Model Accuracy Comparison
model_names = ['Logistic\nRegression', 'Naive\nBayes']
accs = [acc_lr*100, acc_nb*100]
bars = axes[1,0].bar(model_names, accs, color=['#3498DB','#2ECC71'])
axes[1,0].set_title('Model Accuracy Comparison')
axes[1,0].set_ylabel('Accuracy (%)')
axes[1,0].set_ylim(0, 110)
for bar, val in zip(bars, accs):
    axes[1,0].text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+1, f'{val:.1f}%', ha='center', fontweight='bold')

# Plot 5: Rating vs Sentiment
rating_sent = df.groupby(['rating','sentiment']).size().unstack(fill_value=0)
rating_sent.plot(kind='bar', stacked=True, ax=axes[1,1],
                 color=[COLORS.get(c,'grey') for c in rating_sent.columns])
axes[1,1].set_title('Rating vs Sentiment')
axes[1,1].set_xlabel('Rating')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=0)

# Plot 6: Top TF-IDF Words per Sentiment
feature_names = tfidf.get_feature_names_out()
ax = axes[1,2]
ax.set_title('Top Keywords per Sentiment')
y_pos = 0
colors_map = {'positive':'#2ECC71','negative':'#E74C3C','neutral':'#F39C12'}
handles = []
for sentiment in ['positive','negative','neutral']:
    idx = list(y_train).index(sentiment) if sentiment in list(y_train) else None
    mask = y_train == sentiment
    if mask.sum() == 0: continue
    sub_tfidf = X_train_tfidf[mask.values]
    mean_scores = np.asarray(sub_tfidf.mean(axis=0)).flatten()
    top_idx = mean_scores.argsort()[-5:][::-1]
    top_words = [feature_names[i] for i in top_idx]
    top_scores = [mean_scores[i] for i in top_idx]
    for word, score in zip(top_words, top_scores):
        ax.barh(y_pos, score, color=colors_map[sentiment], alpha=0.8)
        ax.text(0.001, y_pos, word, va='center', fontsize=8)
        y_pos += 1
    y_pos += 0.5
    handles.append(mpatches.Patch(color=colors_map[sentiment], label=sentiment))
ax.set_xlabel('Avg TF-IDF Score')
ax.set_yticks([])
ax.legend(handles=handles, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('task4_sentiment_insights.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✓ Chart saved as 'task4_sentiment_insights.png'")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("          FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"✅ Total Reviews Analyzed  : {len(df):,}")
print(f"✅ Positive Reviews        : {(df['sentiment']=='positive').sum():,} ({(df['sentiment']=='positive').mean()*100:.1f}%)")
print(f"✅ Negative Reviews        : {(df['sentiment']=='negative').sum():,} ({(df['sentiment']=='negative').mean()*100:.1f}%)")
print(f"✅ Neutral Reviews         : {(df['sentiment']=='neutral').sum():,}  ({(df['sentiment']=='neutral').mean()*100:.1f}%)")
print(f"✅ Logistic Regression Acc : {acc_lr*100:.2f}%")
print(f"✅ Naive Bayes Accuracy    : {acc_nb*100:.2f}%")
winner = "Logistic Regression" if acc_lr >= acc_nb else "Naive Bayes"
print(f"✅ Best Model              : {winner}")
print("=" * 60)
print("\n✅ TASK 4 COMPLETE!")