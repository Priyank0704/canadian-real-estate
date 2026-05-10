"""
src/utils/config.py

Central configuration for the Canadian Real Estate project.
One place to control all paths, settings, and city lists.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────
    raw_dir: Path       = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path   = Path("reports")
    maps_dir: Path      = Path("reports/maps")
    charts_dir: Path    = Path("reports/charts")

    # ── Cities to analyze ────────────────────────────────────
    # These are the 10 major Canadian markets.
    # CREA uses these exact names in their data.
    cities: list = field(default_factory=lambda: [
        "Toronto",
        "Vancouver",
        "Calgary",
        "Montreal",
        "Ottawa",
        "Edmonton",
        "Hamilton",
        "Winnipeg",
        "Halifax",
        "Victoria",
    ])

    # ── Date range ───────────────────────────────────────────
    start_date: str = "2015-01-01"   # 10 years gives good cycle coverage
    end_date: str   = "2024-12-01"   # Update as new data becomes available

    # ── Forecasting ──────────────────────────────────────────
    forecast_months: int  = 12
    random_seed: int      = 42
    test_months: int      = 12   # holdout for model evaluation

    # ── Affordability index ──────────────────────────────────
    # Median household income by city (StatCan 2021 census, approximate)
    # Used to compute price-to-income ratio
    median_income: dict = field(default_factory=lambda: {
        "Toronto":   85000,
        "Vancouver": 83000,
        "Calgary":   95000,
        "Montreal":  65000,
        "Ottawa":    95000,
        "Edmonton":  88000,
        "Hamilton":  75000,
        "Winnipeg":  72000,
        "Halifax":   68000,
        "Victoria":  78000,
    })

    # ── Map settings ─────────────────────────────────────────
    canada_center_lat: float = 56.0
    canada_center_lon: float = -96.0
    map_zoom: int = 4


cfg = Config()
