# 🔻 Customer Journey Drop-Off Analyzer

An **end-to-end analytics project** that simulates 15,000 e-commerce sessions, maps a 7-stage conversion funnel, identifies where and why users drop off, and presents actionable insights through ML models, A/B testing, RFM segmentation, and an interactive React dashboard.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Interactive Dashboard](#interactive-dashboard)
- [Results & Insights](#results--insights)

---

## Overview

Most e-commerce platforms lose **93 %+** of visitors before they complete a purchase. This project builds a full analytical pipeline to:

1. **Generate** realistic session data with device/UTM/pricing/behavioral attributes
2. **Map** users through a 7-stage funnel (Landing → Browse → Product Detail → Add to Cart → Checkout → Payment → Order Confirmed)
3. **Identify** top drop-off stages and exit reasons (slow load, forced login, unexpected costs, etc.)
4. **Predict** drop-off probability using ML (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
5. **Simulate** an A/B test (guest checkout vs forced login) with statistical significance testing
6. **Segment** customers using RFM analysis (Recency, Frequency, Monetary)
7. **Visualize** everything in an interactive React + Recharts dashboard

---

## Project Structure

```
customer_journey_analyzer/
├── data/
│   ├── raw/
│   │   └── sessions.csv              # 15K synthetic e-commerce sessions
│   └── processed/
│       ├── funnel_summary.csv         # Aggregated funnel stage data
│       ├── rfm_scores.csv             # RFM scores & customer segments
│       └── cohorts.csv                # Monthly cohort retention data
│
├── notebooks/
│   ├── 01_data_generation.ipynb       # Generate synthetic session data
│   ├── 02_eda_and_funnel_analysis.ipynb   # EDA + funnel visualization
│   ├── 03_ml_dropoff_predictor.ipynb  # Train ML models to predict drop-off
│   ├── 04_ab_test_simulation.ipynb    # A/B test: guest checkout experiment
│   ├── 05_cohort_and_rfm_analysis.ipynb   # Cohort retention + RFM segmentation
│   └── 06_plotly_dashboard.ipynb      # Interactive Plotly charts + dashboard link
│
├── src/
│   ├── data_generator.py              # Session simulation engine
│   ├── funnel_analyzer.py             # Funnel building, exit reasons, device/hourly analysis
│   ├── ml_model.py                    # ML pipeline (4 models, SMOTE, SHAP)
│   ├── ab_testing.py                  # A/B test simulation + hypothesis testing
│   ├── cohort_rfm.py                  # Cohort retention + RFM scoring
│   └── visualizer.py                  # Matplotlib & Plotly plotting utilities
│
├── outputs/
│   ├── dashboard_react.html           # ⭐ Interactive React + Recharts dashboard
│   └── figures/                       # Saved chart images (PNG)
│
├── requirements.txt
└── README.md
```

---

## Key Features

### 🔄 Conversion Funnel Analysis
- 7-stage funnel with per-stage drop-off rates and criticality tagging
- Top exit reasons at each stage (e.g., *forced_login* at cart, *unexpected_cost* at checkout)
- Device-level and hourly pattern breakdowns

### 🤖 ML Drop-Off Predictor
- 4 models compared: Logistic Regression, Random Forest, Gradient Boosting, **XGBoost**
- SMOTE for class imbalance handling
- 5-fold stratified cross-validation
- SHAP feature importance analysis
- ROC curve comparison

### 🧪 A/B Test Simulation
- Control vs Treatment (guest checkout enabled)
- 22% recovery rate for forced-login drop-offs
- Two-proportion z-test + Chi-square test
- 95% confidence interval + revenue impact estimation

### 📊 Cohort & RFM Analysis
- Monthly cohort retention heatmap (Jan–Jun 2024)
- RFM scoring (1–5 scale) with 5 customer segments:
  - Champions · Loyal Customers · Potential Loyalists · At Risk · Lost

### 🖥️ Interactive Dashboard
- Self-contained single HTML file (React 18 + Recharts)
- Dark theme with 5 navigable sections
- KPI cards, interactive funnel, retention table, segment charts

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11 |
| **Data** | Pandas, NumPy |
| **ML** | scikit-learn, XGBoost, imbalanced-learn, SHAP |
| **Stats** | SciPy (z-test, chi-square) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | React 18, Recharts, Babel Standalone |
| **Environment** | Jupyter Notebook |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/shrekrr/customer_journey_analyzer.git
cd customer_journey_analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline (notebooks in order):

```bash
jupyter notebook
```

Open and run the notebooks sequentially (`01` → `06`). Each notebook builds on the outputs of the previous one.

### Quick start — generate data only:

```bash
python src/data_generator.py
```

### Open the interactive dashboard:

Simply open `outputs/dashboard_react.html` in any modern browser — no server required.

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `data_generation` | Generates 15K synthetic sessions with realistic drop-off patterns |
| 02 | `eda_and_funnel_analysis` | Builds funnel, identifies exit reasons, analyzes device & hourly patterns |
| 03 | `ml_dropoff_predictor` | Trains 4 ML models with SMOTE + SHAP explainability |
| 04 | `ab_test_simulation` | Simulates guest checkout A/B test with statistical testing |
| 05 | `cohort_and_rfm_analysis` | Monthly cohort retention + RFM customer segmentation |
| 06 | `plotly_dashboard` | Interactive Plotly charts + dashboard HTML verification |

---

## Interactive Dashboard

The React dashboard (`outputs/dashboard_react.html`) includes 5 sections:

| Section | What it shows |
|---------|--------------|
| **Funnel Analysis** | 7-stage funnel with drop-off %, exit reasons, device comparison |
| **ML Predictor** | Model accuracy metrics, feature importance, confusion matrix |
| **A/B Testing** | Conversion rates, statistical significance, revenue impact |
| **Cohort & RFM** | Monthly retention table, RFM segment sizes & revenue |
| **Insights** | Key findings, action plans, top drop-off risks |

---

## Results & Insights

| Metric | Value |
|--------|-------|
| Total Sessions | 15,000 |
| Confirmed Orders | ~1,052 |
| Overall Conversion Rate | ~7.0% |
| Highest Drop-off Stage | Add to Cart → Checkout (43%) |
| Top Exit Reason | Forced login at cart stage |
| A/B Test Result | Guest checkout → statistically significant uplift |
| Best ML Model | XGBoost (highest AUC) |
| Top Predictive Feature | `cart_value`, `load_time_s`, `n_pages_viewed` |

### Key Recommendations

1. **Enable guest checkout** — recovers ~22% of forced-login drop-offs
2. **Optimize page load time** — sessions with >4s load time drop 15% more
3. **Target "At Risk" RFM segment** — 201 customers generating declining revenue
4. **Reduce checkout friction** — unexpected costs drive 43% stage drop-off

---

<p align="center">
  Built with ☕ and Python
</p>
