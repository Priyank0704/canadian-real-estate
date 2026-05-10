"""
src/collection/collect_crea.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from src.utils.config import cfg


CREA_COLUMN_MAP = {
    "Date": "date", "Area": "city", "Average_Price": "avg_price",
    "Median_Price": "median_price", "Sales": "sales_volume",
    "New_Listings": "new_listings", "Active_Listings": "active_listings",
    "SNLR": "snlr", "Months_of_Inventory": "months_inventory",
    "Days_on_Market": "days_on_market", "Composite_HPI": "hpi",
}

CITY_NAME_MAP = {
    "Greater Toronto Area": "Toronto", "Greater Vancouver": "Vancouver",
    "Calgary": "Calgary", "Greater Montreal": "Montreal",
    "Ottawa-Gatineau": "Ottawa", "Edmonton": "Edmonton",
    "Hamilton-Burlington": "Hamilton", "Winnipeg": "Winnipeg",
    "Halifax-Dartmouth": "Halifax", "Victoria": "Victoria",
}


def load_crea_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={k: v for k, v in CREA_COLUMN_MAP.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    if "city" in df.columns:
        df["city"] = df["city"].map(CITY_NAME_MAP).fillna(df["city"])
    df = df[df["city"].isin(cfg.cities)]
    df = df[(df["date"] >= cfg.start_date) & (df["date"] <= cfg.end_date)]
    for col in ["avg_price","median_price","sales_volume","new_listings","active_listings"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["city","date"]).reset_index(drop=True)


def create_sample_crea_data() -> pd.DataFrame:
    """
    Realistic synthetic CREA data using a linear base trajectory.

    FIX: Previous version compounded multipliers monthly causing
    exponential growth (Toronto hit $6M). This version anchors
    each month's price to a linear trajectory between real 2015
    and 2024 price levels, then applies seasonal/event adjustments
    as additive percentage deviations from that trajectory.
    """
    logger.warning("Using SYNTHETIC CREA data — replace with real data for portfolio")

    dates = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="MS")
    n = len(dates)

    city_params = {
        "Toronto":   {"start": 620000,  "end": 1100000, "vol": 0.018, "sales": 8000},
        "Vancouver": {"start": 800000,  "end": 1200000, "vol": 0.020, "sales": 3000},
        "Calgary":   {"start": 450000,  "end": 570000,  "vol": 0.022, "sales": 2500},
        "Montreal":  {"start": 320000,  "end": 550000,  "vol": 0.015, "sales": 4000},
        "Ottawa":    {"start": 380000,  "end": 650000,  "vol": 0.016, "sales": 1800},
        "Edmonton":  {"start": 380000,  "end": 430000,  "vol": 0.020, "sales": 2000},
        "Hamilton":  {"start": 400000,  "end": 820000,  "vol": 0.022, "sales": 1200},
        "Winnipeg":  {"start": 280000,  "end": 360000,  "vol": 0.014, "sales": 1000},
        "Halifax":   {"start": 250000,  "end": 480000,  "vol": 0.016, "sales": 600},
        "Victoria":  {"start": 550000,  "end": 880000,  "vol": 0.018, "sales": 400},
    }

    np.random.seed(cfg.random_seed)
    records = []

    for city, p in city_params.items():
        base_trajectory = np.linspace(p["start"], p["end"], n)

        for i, date in enumerate(dates):
            base_price = base_trajectory[i]
            month = date.month
            seasonal_pct = 0.04 * np.sin(2 * np.pi * (month - 3) / 12)

            if pd.Timestamp("2020-07-01") <= date <= pd.Timestamp("2021-12-01"):
                event_pct = 0.12
            elif pd.Timestamp("2022-04-01") <= date <= pd.Timestamp("2023-06-01"):
                months_in = (date - pd.Timestamp("2022-04-01")).days / 30
                event_pct = -min(0.18, months_in * 0.012)
            elif date >= pd.Timestamp("2023-07-01"):
                months_rec = (date - pd.Timestamp("2023-07-01")).days / 30
                event_pct = min(0.06, months_rec * 0.008)
            else:
                event_pct = 0.0

            noise_pct = np.random.normal(0, p["vol"])
            price = base_price * (1 + seasonal_pct + event_pct + noise_pct)
            price = max(price, p["start"] * 0.70)

            sales = p["sales"] * (1 + seasonal_pct) * (0.85 + 0.3 * np.random.random())
            if pd.Timestamp("2022-04-01") <= date <= pd.Timestamp("2023-06-01"):
                sales *= 0.65
            new_listings = sales * (1.2 + 0.4 * np.random.random())

            records.append({
                "date":             date,
                "city":             city,
                "avg_price":        round(price),
                "median_price":     round(price * 0.92),
                "sales_volume":     round(max(sales, 10)),
                "new_listings":     round(max(new_listings, 15)),
                "active_listings":  round(max(new_listings * 2.2, 30)),
                "days_on_market":   round(15 + 10 * np.random.random() +
                                          (20 if event_pct < -0.05 else 0), 1),
                "months_inventory": round(2 + 3 * np.random.random() +
                                          (2 if event_pct < -0.05 else 0), 1),
            })

    df = pd.DataFrame(records)
    df["snlr"] = (df["sales_volume"] / df["new_listings"]).clip(0.1, 1.5)
    logger.info(f"Synthetic CREA data created: {df.shape}")
    return df


def load_or_create_crea() -> pd.DataFrame:
    crea_path = cfg.raw_dir / "crea" / "crea_mls_stats.csv"
    if crea_path.exists():
        return load_crea_csv(str(crea_path))
    logger.warning(
        f"CREA CSV not found at {crea_path}. "
        "Using synthetic data. Download from https://creastats.crea.ca/"
    )
    return create_sample_crea_data()


if __name__ == "__main__":
    df = load_or_create_crea()
    print(df.groupby("city")[["avg_price"]].agg(["min", "max"]))
