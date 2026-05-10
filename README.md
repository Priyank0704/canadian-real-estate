# 🏠 Canadian Real Estate Market Intelligence

An end-to-end data science project analyzing Canadian housing markets across 10 major cities — combining CREA MLS statistics, Bank of Canada interest rates, and Statistics Canada data into an interactive intelligence dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://canadian-real-estate-7rnzxbgf2lw7octpe8jdz9.streamlit.app/)

---

## Live Demo

| Resource | Link |
|---|---|
| **Live Dashboard** | https://canadian-real-estate-7rnzxbgf2lw7octpe8jdz9.streamlit.app/ |
| **GitHub** | https://github.com/Priyank0704/canadian-real-estate |

---

## Project Overview

This project tells the data story of Canadian housing from 2015 to 2024 — including the fastest interest rate hiking cycle in Canadian history (0.25% → 5.00% between March 2022 and July 2023) and its devastating impact on housing affordability across major markets.

**Why this project matters for Canadian employers:**
Most data science portfolios use US or global datasets. This project uses Canadian-specific data sources (CREA, Bank of Canada, Statistics Canada), Canadian city geographies, and analyzes events that every Canadian employer recognizes — the housing affordability crisis and the 2022 rate hiking cycle.

---

## Key Findings

- **Vancouver** is the least affordable city at **15.3x annual income** (global threshold: 5x = severely unaffordable)
- **Edmonton** is the most affordable major market at **4.9x annual income**
- Toronto prices dropped **~16%** peak-to-trough during the 2022–2023 rate hiking cycle
- BoC rate and Toronto housing prices show a **-0.72 correlation** — strong negative relationship
- **SARIMA** outperforms XGBoost on stable markets (Toronto, Hamilton); **XGBoost** wins on dynamic markets (Edmonton, Calgary)

---

## Dashboard Features

### Tab 1 — National Overview
- KPI cards: cities tracked, highest price, most/least affordable, avg days on market
- Interactive map with dropdown to switch between Average Price, Affordability Ratio, and YoY Change

### Tab 2 — City Deep Dive
- Select any of 10 cities from the sidebar
- Price history chart with rate hiking period highlighted
- Affordability ratio over time vs the 5x global threshold
- 12-month price forecast with uncertainty intervals
- Model selector (SARIMA / XGBoost)

### Tab 3 — The Rate Hike Story
- Dual-axis chart: BoC rate vs Toronto price (2015–2024)
- Correlation statistics and peak-to-trough analysis
- The clearest visualization of how monetary policy affects housing

### Tab 4 — Model Comparison
- MAPE table coloured green (accurate) to red (less accurate)
- Bar chart comparing SARIMA vs XGBoost across all 10 cities
- Model selection rationale by market type

---

## Architecture

```
Data Collection          Feature Engineering       Modelling
─────────────────        ───────────────────       ─────────────────
CREA MLS Stats    ──┐    Time features             SARIMA
Bank of Canada    ──┼──► Card aggregations    ──►  XGBoost (lag features)
Statistics Canada ──┘    Affordability index       12-month forecast
                         Market heat (SNLR)        MAPE evaluation

Visualization            Dashboard
─────────────────        ─────────────────
Folium maps       ──┐    Streamlit (5 tabs)
Plotly charts     ──┼──► Streamlit Cloud deployment
Seasonal decomp   ──┘    Interactive city selector
```

---

## Repository Structure

```
canadian-real-estate/
├── src/
│   ├── collection/
│   │   ├── collect_crea.py       # CREA housing statistics loader
│   │   ├── collect_boc.py        # Bank of Canada rate downloader
│   │   └── build_dataset.py      # Merges all 3 sources → master dataset
│   ├── features/
│   │   └── eda.py                # 7 interactive EDA charts
│   ├── models/
│   │   └── forecast.py           # SARIMA + XGBoost forecasting pipeline
│   ├── visualization/
│   │   └── maps.py               # 5 Folium/Plotly interactive maps
│   ├── dashboard/
│   │   └── app.py                # Streamlit dashboard (5 tabs)
│   └── utils/
│       ├── config.py             # Central configuration
│       └── data_loader.py        # Data I/O utilities
├── requirements.txt
├── packages.txt                  # System dependencies for Streamlit Cloud
├── runtime.txt                   # Python version specification
└── README.md
```

---

## Data Sources

| Source | Data | URL |
|---|---|---|
| CREA | Monthly MLS housing stats (avg price, sales, listings) | creastats.crea.ca |
| Bank of Canada | Overnight rate history (Valet API) | bankofcanada.ca/rates |
| Statistics Canada | Median household income by city | 150.statcan.gc.ca |

> **Note:** Real CREA data requires free registration at creastats.crea.ca. The project includes a realistic synthetic dataset for immediate use that mirrors real market dynamics including the COVID surge and 2022 rate correction.

---

## Quick Start

```bash
# Clone
git clone https://github.com/Priyank0704/canadian-real-estate.git
cd canadian-real-estate

# Setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Build data pipeline
python -m src.collection.build_dataset

# Run EDA
python -m src.features.eda

# Build maps (view via: python -m http.server 8080)
python -m src.visualization.maps

# Train forecast models
python -m src.models.forecast

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Model Performance (MAPE — lower is better)

| City | SARIMA | XGBoost | Best |
|---|---|---|---|
| Toronto | 4.4% | 10.5% | SARIMA |
| Edmonton | 6.9% | 3.2% | XGBoost |
| Calgary | 6.2% | 5.0% | XGBoost |
| Montreal | 9.5% | 8.6% | XGBoost |
| Ottawa | 8.9% | 9.8% | SARIMA |
| Hamilton | 9.3% | 15.6% | SARIMA |
| Winnipeg | 8.8% | 5.7% | XGBoost |
| Halifax | 6.7% | 12.4% | SARIMA |
| Victoria | 9.4% | 8.3% | XGBoost |
| Vancouver | 9.2% | 8.0% | XGBoost |

**Insight:** SARIMA wins on stable, lower-volatility markets. XGBoost wins on dynamic markets with stronger lag patterns. This confirms that model selection should be market-specific, not one-size-fits-all.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data processing | pandas, numpy, scipy |
| Time series | statsmodels (SARIMA) |
| Machine learning | XGBoost, scikit-learn, Optuna |
| Geospatial | Folium, GeoPandas |
| Visualisation | Plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Author

**Priyank** — Post Graduate in Big Data Analytics (2024) and Artificial Intelligence (2025)

- GitHub: [@Priyank0704](https://github.com/Priyank0704)
- Project 1: [Fraud Detection MLOps Pipeline](https://github.com/Priyank0704/fraud-mlops)
