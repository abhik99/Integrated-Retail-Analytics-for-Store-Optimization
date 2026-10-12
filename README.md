# 🏬 Integrated Retail Analytics for Store Optimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![HistGradientBoosting](https://img.shields.io/badge/Algorithm-HistGradientBoosting-blue?style=for-the-badge)](https://scikit-learn.org)
[![K-Means](https://img.shields.io/badge/Clustering-K--Means%20%26%20PCA-success?style=for-the-badge)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

A comprehensive retail predictive analytics and demand forecasting system covering 45 department stores, 81 product departments, and 143 weeks (421,570 transaction records). Integrates seasonal markdown elasticity, macroeconomic indicators (unemployment, CPI, fuel price volatility), store clustering, and high-performance gradient boosting to optimize departmental replenishment.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Exploratory Analysis & Macro Insights](#-exploratory-analysis--macro-insights)
- [Model Evaluation & Metrics](#-model-evaluation--metrics)
- [Store Clustering & Segmentation](#-store-clustering--segmentation)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Author & Connect](#-author--connect)

---

## 🚀 Executive Overview
Modern retailers face high margin volatility: overstocking leads to aggressive clearance markdowns, while understocking during peak holiday seasons causes permanent revenue loss.

**Integrated Retail Analytics** models sales across 421,570 departmental records using:
- **Multimodal Datasets:** Merged store physical profiles (`stores data-set.csv`), weekly departmental sales (`sales data-set.csv`), and external features (`Features data set.csv`).
- **Calendar & Holiday Interactions:** Dedicated tracking of Super Bowl, Labor Day, Thanksgiving, and Christmas surge dynamics.
- **Promotional Elasticity:** Impact analysis of five anonymized promotional markdowns introduced in November 2011.
- **Production Pipeline:** Serialized preprocessor (`model_feature.pkl`) and tuned forecasting engine (`best_sales_model.pkl`).

---

## 🏗️ System Architecture

```text
[ Data Ingestion: Stores (45) | Sales (421k) | Features (CPI, Markdowns) ]
                                   │
                                   ▼
┌────────────────────────────────────────────────────────┐
│            Feature Engineering & Preprocessing         │
│   • Date parsing & temporal indicators (Week, DayOfYear)│
│   • Total_MarkDown aggregation across 5 campaigns      │
│   • Right-skew target handling (log1p transform)       │
│   • LabelEncoding Store Type (A, B, C)                 │
└────────────────────────────────────────────────────────┘
                                   │
       ┌───────────────────────────┴───────────────────────────┐
       ▼                                                       ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│ Unsupervised Store Segmentation │   │  Supervised Sales Forecasting   │
│   • StandardScaler normalization │   │   • Linear Regression Baseline  │
│   • PCA 2-D Dimensionality Red. │   │   • Lightweight Random Forest   │
│   • K-Means Store Clustering    │   │   • HistGradientBoostingRegressor│
└─────────────────────────────────┘   └─────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────┐
│             Evaluation on Business Metrics             │
│   • WMAE (Weighted Mean Absolute Error - 5x Holidays)  │
│   • RMSE & Dollar-Denominated Inventory Penalties      │
└────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Features
- **Holiday-Weighted Error Metric (WMAE):** Penalizes forecasting errors during holiday weeks 5x more severely than regular weeks, aligning ML performance directly with peak retail risk.
- **Macroeconomic Correlation Analysis:** Evaluates whether Consumer Price Index (CPI), regional unemployment, and fuel prices exhibit linear or non-linear effects on consumer spending.
- **Store Segmentation (K-Means + PCA):** Classifies store formats into distinct clusters (mega-centers vs suburban vs urban express) to customize replenishment schedules.
- **Production Pickling:** Serialized model pipeline ready for weekly batch inference.

---

## 🧠 Model Evaluation & Metrics

| Model | WMAE (Holiday Weighted) | RMSE ($) | R² Score | Key Takeaway |
|---|---|---|---|---|
| **Hist Gradient Boosting (Best)** | **$1,620** | **$4,310** | **0.932** | **Fastest training, best non-linear markdown handling** |
| Random Forest Regressor | $1,680 | $4,420 | 0.926 | Strong non-linear interactions, higher memory footprint |
| Ridge Regularized Linear | $9,850 | $14,120 | 0.384 | Too rigid for complex department-level fluctuations |
| Linear Regression Baseline | $9,890 | $14,190 | 0.381 | Fails on multi-variable departmental crosses |

---

## 💡 Store Clustering & Segmentation
- **Cluster 1 (Large Format / Type A):** Massive square footage, highest total sales volume, extreme holiday spikes, highly responsive to MarkDown1 and MarkDown3.
- **Cluster 2 (Medium Suburban / Type B):** Steady weekly baseline, moderate markdown sensitivity, resilient to fuel price shocks.
- **Cluster 3 (Compact Express / Type C):** Small footprint, localized grocery and essentials demand, minimal markdown interaction.

---

## 📁 Repository Structure
```text
├── Integrated_Retail_Analytics.ipynb  # Comprehensive EDA, hypothesis testing & ML notebook
├── Features data set.csv              # Macroeconomic indicators & promotional markdowns
├── sales data-set.csv                 # 421,570 weekly department sales transactions
├── stores data-set.csv                # Store metadata (Type A/B/C, Size)
├── best_sales_model.pkl               # Serialized HistGradientBoosting Regressor
├── model_feature.pkl                  # Serialized feature processing transformer
├── requirments.txt                    # Environment dependencies
└── README.md                          # Project documentation
```

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/abhik99/Integrated-Retail-Analytics-for-Store-Optimization.git
cd Integrated-Retail-Analytics-for-Store-Optimization
```

### 2. Install Dependencies
```bash
pip install -r requirments.txt
```

### 3. Run Inference on Unseen Records
```python
import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load('best_sales_model.pkl')
feature_pipeline = joblib.load('model_feature.pkl')

# Predict weekly sales
# sample_input = pd.DataFrame(...)
# predictions = model.predict(sample_input)
```

---

## 👨‍💻 Author & Connect

**Abhishek Kumar (Abhi)**  
*Data Scientist & Analytics Engineer*

- 🌐 **Portfolio:** [abhicodes.online](https://abhicodes.online)
- 💼 **LinkedIn:** [linkedin.com/in/abhik99](https://linkedin.com/in/abhik99)
- 🐙 **GitHub:** [@abhik99](https://github.com/abhik99)
- 📧 **Email:** [akdevil27boy@gmail.com](mailto:akdevil27boy@gmail.com)

---
*Distributed under the MIT License.*
