"""
src/collection/build_dataset.py

Merges all 3 data sources into one clean master dataset.

Usage (from project root):
    python -m src.collection.build_dataset

Output:
    data/processed/master_dataset.parquet
    data/processed/city_monthly.parquet

WHY MERGE EVERYTHING HERE:
    All downstream steps (EDA, maps, forecasting, dashboard) use
    the same clean master dataset. One source of truth.
    If we change data collection, we re-run this script and
    everything downstream automatically gets the update.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from src.utils.config import cfg
from src.collection.collect_crea import load_or_create_crea
from src.collection.collect_boc import load_or_create_boc


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the master dataset.

    WHY HERE (not in a separate features module):
        These features are derived from the raw data columns and
        are needed by ALL downstream components — EDA, maps, models.
        They're part of the dataset definition, not model-specific.
    """
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # ── Price momentum features ───────────────────────────────
    df["price_mom_1m"] = df.groupby("city")["avg_price"].pct_change(1) * 100
    df["price_mom_3m"] = df.groupby("city")["avg_price"].pct_change(3) * 100
    df["price_mom_12m"] = df.groupby("city")["avg_price"].pct_change(12) * 100  # YoY

    # ── Price moving averages ─────────────────────────────────
    df["price_ma3"]  = df.groupby("city")["avg_price"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["price_ma12"] = df.groupby("city")["avg_price"].transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )

    # ── Affordability index ───────────────────────────────────
    # Price-to-annual-income ratio. Higher = less affordable.
    # A ratio > 5 is generally considered unaffordable.
    df["income"] = df["city"].map(cfg.median_income)
    df["affordability_ratio"] = df["avg_price"] / df["income"]

    # ── Market heat indicator ─────────────────────────────────
    # SNLR > 0.6 = seller's market, < 0.4 = buyer's market
    if "snlr" not in df.columns:
        df["snlr"] = df["sales_volume"] / df["new_listings"].replace(0, np.nan)
    df["market_type"] = pd.cut(
        df["snlr"],
        bins=[0, 0.4, 0.6, 100],
        labels=["buyers", "balanced", "sellers"]
    )

    # ── Rate change features ──────────────────────────────────
    if "boc_rate" in df.columns:
        df["rate_change_1m"] = df["boc_rate"].diff(1)
        df["rate_change_3m"] = df["boc_rate"].diff(3)
        df["rate_hike_cycle"] = (
            (df["boc_rate"] >= 2.0) & (df["rate_change_3m"] > 0)
        ).astype(int)

    # ── Calendar features (useful for forecasting) ────────────
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    logger.info(f"Features engineered. Final shape: {df.shape}")
    return df


def build_master_dataset() -> pd.DataFrame:
    """
    Main function: load, merge, and save the master dataset.
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all sources ──────────────────────────────────────
    logger.info("Loading data sources...")
    crea_df = load_or_create_crea()
    boc_df  = load_or_create_boc()

    logger.info(f"CREA shape:  {crea_df.shape}")
    logger.info(f"BoC shape:   {boc_df.shape}")

    # ── Merge CREA + BoC on date ──────────────────────────────
    # WHY LEFT JOIN on CREA:
    #   We want all housing data rows. BoC rate is added where available.
    #   The BoC rate is then forward-filled for months with no change
    #   (rates only change on specific announcement dates).
    master = crea_df.merge(boc_df, on="date", how="left")
    master["boc_rate"] = master["boc_rate"].ffill().bfill()

    # ── Add StatCan income data (from config) ─────────────────
    # In a full project you'd download this from StatCan API.
    # We use the config values which are based on 2021 census.
    master["median_income"] = master["city"].map(cfg.median_income)

    # ── Engineer features ─────────────────────────────────────
    master = engineer_features(master)

    # ── Save master dataset ───────────────────────────────────
    master_path = cfg.processed_dir / "master_dataset.parquet"
    master.to_parquet(master_path, index=False)
    logger.info(f"Master dataset saved → {master_path} ({master.shape})")

    # ── Save city-level summary ───────────────────────────────
    # Latest snapshot per city — used by the dashboard and maps
    latest = (
        master.sort_values("date")
        .groupby("city")
        .last()
        .reset_index()
    )
    latest_path = cfg.processed_dir / "city_latest.parquet"
    latest.to_parquet(latest_path, index=False)
    logger.info(f"City latest snapshot saved → {latest_path}")

    # ── Print summary ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    for city in cfg.cities:
        city_data = master[master["city"] == city]
        if len(city_data) > 0:
            latest_price = city_data["avg_price"].iloc[-1]
            yoy = city_data["price_mom_12m"].iloc[-1]
            ratio = city_data["affordability_ratio"].iloc[-1]
            logger.info(
                f"{city:<12} | Price: ${latest_price:>10,.0f} | "
                f"YoY: {yoy:>+6.1f}% | Affordability ratio: {ratio:.1f}x"
            )
    logger.info("=" * 60)
    logger.info("Milestone 1 complete. Run Milestone 2 next:")
    logger.info("  python -m src.features.eda")
    logger.info("=" * 60)

    return master


if __name__ == "__main__":
    build_master_dataset()
