#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 16:45:12 2025

@author: ozancelayir
"""

"""
===========================================================
Machine Learning Web Analytics Project - RetailRocket Dataset
===========================================================

🔍 Project Purpose:
Build an end-to-end machine learning pipeline using RetailRocket ecommerce data to generate actionable business insights.

📦 Dataset Description:
- events.csv: User interactions (view, cart, purchase) with timestamps.
- category_tree.csv: Product categories hierarchy.
- item_properties_part1.csv & item_properties_part2.csv: Product properties over time.

🗺️ Script Structure:
1. Data Loading & Preparation
2. Exploratory Data Analysis (EDA)
3. Unsupervised Learning - Customer Segmentation
4. Supervised Learning - Purchase Prediction
5. Time-Series Analysis - Transaction Trend Analysis
6. PDF Report Generation

▶️ How to Run:
Ensure the RetailRocket dataset CSV files are in a ./data/ folder relative to this script.
Run the script using any Python IDE or terminal:
    python retailrocket_ml_pdf.py

Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, fpdf

===========================================================
"""

# =============================
# 📦 Import Libraries
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from fpdf import FPDF
import os
import warnings
warnings.filterwarnings('ignore')

# =============================
# 🔗 Data Loading & Preparation
# =============================

# Create output folder
os.makedirs('./output', exist_ok=True)

# Load datasets
events = pd.read_csv('./data/events.csv')
category_tree = pd.read_csv('./data/category_tree.csv')
item_props1 = pd.read_csv('./data/item_properties_part1.csv')
item_props2 = pd.read_csv('./data/item_properties_part2.csv')

# Merge item properties
item_properties = pd.concat([item_props1, item_props2])

# Convert timestamps
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

# =============================
# 📊 Exploratory Data Analysis
# =============================

# Plot 1: Event type distribution
plt.figure(figsize=(6,4))
sns.countplot(data=events, x='event')
plt.title('Event Type Distribution')
plt.tight_layout()
plt.savefig('./output/event_distribution.png')
plt.close()

# Plot 2: Top viewed items
top_items = events[events['event'] == 'view'].itemid.value_counts().head(10)
plt.figure(figsize=(6,4))
top_items.plot(kind='bar', title='Top 10 Viewed Items')
plt.tight_layout()
plt.savefig('./output/top_viewed_items.png')
plt.close()

# Plot 3: User activity distribution
user_activity = events.groupby('visitorid').size()
plt.figure(figsize=(6,4))
sns.histplot(user_activity, bins=50)
plt.title('User Activity Distribution')
plt.xlabel('Number of Events per User')
plt.tight_layout()
plt.savefig('./output/user_activity_distribution.png')
plt.close()

# =============================
# 🔍 Unsupervised Learning
# =============================

# Create user-level features
user_features = events.pivot_table(index='visitorid', columns='event', aggfunc='size', fill_value=0).reset_index()

# Ensure all event types exist
for col in ['view', 'cart', 'transaction']:
    if col not in user_features.columns:
        user_features[col] = 0

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_features[['view', 'cart', 'transaction']])

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
user_features['cluster'] = kmeans.fit_predict(X_scaled)

# Plot 4: Customer Segments
plt.figure(figsize=(6,4))
sns.scatterplot(x=user_features['view'], y=user_features['transaction'], hue=user_features['cluster'], palette='Set1')
plt.title('Customer Segments')
plt.tight_layout()
plt.savefig('./output/customer_segments.png')
plt.close()

# =============================
# 🎯 Supervised Learning
# =============================

# Session-level data
session_data = events.groupby(['visitorid', 'itemid']).event.agg(['count']).reset_index()

# Label: whether there was a purchase
purchase = events[events['event'] == 'transaction']
session_data['purchase'] = session_data.set_index(['visitorid','itemid']).index.isin(
    purchase.set_index(['visitorid','itemid']).index).astype(int)

# Features and target
X = session_data[['count']]
y = session_data['purchase']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Purchase Prediction')
plt.tight_layout()
plt.savefig('./output/confusion_matrix.png')
plt.close()

# =============================
# ⏳ Time-Series Analysis
# =============================

# Daily transaction counts
transactions = events[events['event'] == 'transaction']
daily_tx = transactions.groupby(transactions['timestamp'].dt.date).size()

# Plot 6: Transaction trend
plt.figure(figsize=(8,4))
daily_tx.plot(title='Daily Transaction Trend')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.savefig('./output/daily_transaction_trend.png')
plt.close()

# Decomposition
decomposition = seasonal_decompose(daily_tx, model='additive', period=7)
decomposition.plot()
plt.tight_layout()
plt.savefig('./output/seasonal_decomposition.png')
plt.close()

# ARIMA
model_arima = ARIMA(daily_tx, order=(1,1,1))
model_fit = model_arima.fit()
forecast = model_fit.forecast(steps=7)

# Plot 8: ARIMA forecast
plt.figure(figsize=(8,4))
plt.plot(daily_tx, label='Historical')
plt.plot(pd.date_range(daily_tx.index[-1], periods=7, freq='D'), forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast - Next 7 Days')
plt.legend()
plt.tight_layout()
plt.savefig('./output/arima_forecast.png')
plt.close()

