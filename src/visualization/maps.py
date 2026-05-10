"""
src/visualization/maps.py

Builds interactive choropleth and marker maps of Canadian housing data.

WHY FOLIUM (not Plotly maps):
    Folium wraps Leaflet.js — the industry standard for interactive
    web maps. It produces standalone HTML files with full zoom,
    pan, hover tooltips, and layer controls. No API key needed.
    The output embeds directly in the Streamlit dashboard.

Maps generated (saved to reports/maps/):
    01_avg_price_map.html        - City markers sized/coloured by avg price
    02_affordability_map.html    - Choropleth by affordability ratio
    03_yoy_change_map.html       - YoY price change (green=up, red=down)
    04_market_type_map.html      - Seller vs buyer vs balanced markets

Usage:
    python -m src.visualization.maps
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
from pathlib import Path
from loguru import logger

from src.utils.config import cfg


# ── City coordinates ──────────────────────────────────────────────────────────
# Approximate lat/lon for each city centre
CITY_COORDS = {
    "Toronto":   (43.6532, -79.3832),
    "Vancouver": (49.2827, -123.1207),
    "Calgary":   (51.0447, -114.0719),
    "Montreal":  (45.5017, -73.5673),
    "Ottawa":    (45.4215, -75.6972),
    "Edmonton":  (53.5461, -113.4938),
    "Hamilton":  (43.2557, -79.8711),
    "Winnipeg":  (49.8951, -97.1384),
    "Halifax":   (44.6488, -63.5752),
    "Victoria":  (48.4284, -123.3656),
}


def load_latest() -> pd.DataFrame:
    """Load the latest snapshot per city."""
    path = cfg.processed_dir / "city_latest.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "city_latest.parquet not found. "
            "Run: python -m src.collection.build_dataset"
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    # Add coordinates
    df["lat"] = df["city"].map(lambda c: CITY_COORDS.get(c, (0, 0))[0])
    df["lon"] = df["city"].map(lambda c: CITY_COORDS.get(c, (0, 0))[1])
    df = df[df["lat"] != 0]   # drop cities without coords
    logger.info(f"Loaded latest snapshot: {df.shape}")
    return df


def load_master() -> pd.DataFrame:
    """Load full historical dataset."""
    path = cfg.processed_dir / "master_dataset.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Helper: color scales ──────────────────────────────────────────────────────

def price_to_color(price: float,
                   min_p: float, max_p: float) -> str:
    """Map a price to a blue gradient hex color."""
    ratio = (price - min_p) / (max_p - min_p + 1)
    # Blue gradient: light blue → dark blue
    r = int(210 - ratio * 160)
    g = int(230 - ratio * 160)
    b = int(255 - ratio * 50)
    return f"#{r:02x}{g:02x}{b:02x}"


def affordability_to_color(ratio: float) -> str:
    """
    Map affordability ratio to color.
    < 5:   green  (affordable)
    5-8:   yellow (moderate)
    8-12:  orange (unaffordable)
    > 12:  red    (severely unaffordable)
    """
    if ratio < 5:
        return "#3B6D11"   # green
    elif ratio < 8:
        return "#BA7517"   # amber
    elif ratio < 12:
        return "#D85A30"   # orange
    else:
        return "#A32D2D"   # dark red


def yoy_to_color(yoy: float) -> str:
    """Green for positive YoY, red for negative."""
    if pd.isna(yoy):
        return "#888888"
    elif yoy > 10:
        return "#27500A"
    elif yoy > 5:
        return "#3B6D11"
    elif yoy > 0:
        return "#639922"
    elif yoy > -5:
        return "#E24B4A"
    else:
        return "#A32D2D"


def radius_from_price(price: float,
                      min_p: float, max_p: float,
                      min_r: int = 8, max_r: int = 30) -> int:
    """Scale circle radius proportionally to price."""
    ratio = (price - min_p) / (max_p - min_p + 1)
    return int(min_r + ratio * (max_r - min_r))


def make_base_map(zoom: int = 4) -> folium.Map:
    """Create a base Folium map centered on Canada."""
    return folium.Map(
        location=[cfg.canada_center_lat, cfg.canada_center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",   # clean, light basemap
        attr="CartoDB",
    )


# ── Map 1: Average price circles ─────────────────────────────────────────────

def map_avg_price(df: pd.DataFrame):
    """
    Circle markers sized and coloured by average price.

    WHY CIRCLE MARKERS (not pins):
        Pins all look identical — you can't see differences at a glance.
        Circle markers encode TWO dimensions simultaneously:
        - Size  → price magnitude (Vancouver circle is biggest)
        - Color → price level (darker blue = higher price)
        This is more information-dense and visually clearer.
    """
    m = make_base_map()
    min_p, max_p = df["avg_price"].min(), df["avg_price"].max()

    for _, row in df.iterrows():
        color  = price_to_color(row["avg_price"], min_p, max_p)
        radius = radius_from_price(row["avg_price"], min_p, max_p)

        # Popup with full details (shown on click)
        popup_html = f"""
        <div style="font-family:Arial;min-width:200px">
          <h4 style="margin:0 0 8px;color:#185FA5">{row['city']}</h4>
          <table style="width:100%;font-size:13px">
            <tr><td><b>Avg Price</b></td>
                <td style="text-align:right">${row['avg_price']:,.0f}</td></tr>
            <tr><td><b>Median Price</b></td>
                <td style="text-align:right">${row['median_price']:,.0f}</td></tr>
            <tr><td><b>YoY Change</b></td>
                <td style="text-align:right;color:{'green' if row.get('price_mom_12m',0) > 0 else 'red'}">
                {row.get('price_mom_12m', 0):+.1f}%</td></tr>
            <tr><td><b>Affordability</b></td>
                <td style="text-align:right">{row.get('affordability_ratio', 0):.1f}x income</td></tr>
            <tr><td><b>Days on Market</b></td>
                <td style="text-align:right">{row.get('days_on_market', 0):.0f} days</td></tr>
          </table>
        </div>
        """

        # Tooltip shown on hover (brief)
        tooltip = (f"{row['city']}: ${row['avg_price']:,.0f} "
                   f"({row.get('price_mom_12m', 0):+.1f}% YoY)")

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=tooltip,
        ).add_to(m)

    # Add a simple legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-family:Arial;font-size:13px">
      <b>Average Price</b><br>
      <span style="color:#d2e6ff">&#9679;</span> Lower price<br>
      <span style="color:#5282d4">&#9679;</span> Mid range<br>
      <span style="color:#183064">&#9679;</span> Higher price<br>
      <i style="font-size:11px;color:#888">Circle size = relative price</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = cfg.maps_dir / "01_avg_price_map.html"
    m.save(str(path))
    logger.info(f"Saved: {path}")


# ── Map 2: Affordability choropleth ──────────────────────────────────────────

def map_affordability(df: pd.DataFrame):
    """
    Circle markers coloured by affordability ratio.

    This is the most analytically powerful map — it shows at a glance
    which cities are liveable for average earners vs which are
    effectively closed to median-income Canadians.
    """
    m = make_base_map()

    for _, row in df.iterrows():
        ratio  = row.get("affordability_ratio", 0)
        color  = affordability_to_color(ratio)
        radius = 12 + ratio * 1.2   # bigger circle = less affordable

        popup_html = f"""
        <div style="font-family:Arial;min-width:220px">
          <h4 style="margin:0 0 8px;color:#185FA5">{row['city']}</h4>
          <p style="margin:0 0 6px;font-size:13px">
            <b>Affordability ratio: {ratio:.1f}x</b><br>
            Annual income needed: ${row['avg_price']/ratio:,.0f}<br>
            Avg home price: ${row['avg_price']:,.0f}
          </p>
          <div style="padding:6px;border-radius:4px;font-size:12px;
                      background:{'#EAF3DE' if ratio < 5 else '#FAEEDA' if ratio < 8 else '#FCEBEB'};
                      color:{'#27500A' if ratio < 5 else '#633806' if ratio < 8 else '#7A1010'}">
            {'✓ Relatively affordable' if ratio < 5 else
             '⚠ Moderately unaffordable' if ratio < 8 else
             '✗ Severely unaffordable'}
          </div>
        </div>
        """
        tooltip = f"{row['city']}: {ratio:.1f}x annual income"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=tooltip,
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-family:Arial;font-size:13px">
      <b>Affordability Ratio</b><br>
      <span style="color:#3B6D11">&#9679;</span> &lt;5x — Affordable<br>
      <span style="color:#BA7517">&#9679;</span> 5–8x — Moderate<br>
      <span style="color:#D85A30">&#9679;</span> 8–12x — Unaffordable<br>
      <span style="color:#A32D2D">&#9679;</span> &gt;12x — Severely unaffordable<br>
      <i style="font-size:11px;color:#888">Global threshold: 5x = severe</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = cfg.maps_dir / "02_affordability_map.html"
    m.save(str(path))
    logger.info(f"Saved: {path}")


# ── Map 3: Year-over-year change ──────────────────────────────────────────────

def map_yoy_change(df: pd.DataFrame):
    """
    Circle markers coloured by YoY price change.
    Green = prices rising, Red = prices falling.
    Shows which markets are heating up vs cooling down right now.
    """
    m = make_base_map()

    for _, row in df.iterrows():
        yoy   = row.get("price_mom_12m", 0)
        color = yoy_to_color(yoy)

        popup_html = f"""
        <div style="font-family:Arial;min-width:200px">
          <h4 style="margin:0 0 8px">{row['city']}</h4>
          <p style="font-size:13px;margin:0">
            <b>YoY change: {yoy:+.1f}%</b><br>
            Current price: ${row['avg_price']:,.0f}<br>
            3-month change: {row.get('price_mom_3m', 0):+.1f}%
          </p>
        </div>
        """
        tooltip = f"{row['city']}: {yoy:+.1f}% YoY"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=14,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=tooltip,
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-family:Arial;font-size:13px">
      <b>Year-over-Year Price Change</b><br>
      <span style="color:#27500A">&#9679;</span> &gt;10% — Strong growth<br>
      <span style="color:#3B6D11">&#9679;</span> 5–10% — Moderate growth<br>
      <span style="color:#639922">&#9679;</span> 0–5% — Slight growth<br>
      <span style="color:#E24B4A">&#9679;</span> 0 to -5% — Declining<br>
      <span style="color:#A32D2D">&#9679;</span> &lt;-5% — Sharp decline
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = cfg.maps_dir / "03_yoy_change_map.html"
    m.save(str(path))
    logger.info(f"Saved: {path}")


# ── Map 4: Market type (buyer/balanced/seller) ────────────────────────────────

def map_market_type(df: pd.DataFrame):
    """
    Circle markers coloured by market type (SNLR-based).
    Shows which cities currently favour buyers vs sellers.

    WHY THIS IS USEFUL:
        This is a forward-looking signal. Seller's markets predict
        future price increases; buyer's markets predict future declines.
        A hiring manager at a real estate firm or mortgage company
        would immediately understand and value this chart.
    """
    m = make_base_map()

    market_colors = {
        "sellers":  "#27500A",   # green — sellers have power
        "balanced": "#BA7517",   # amber — equal footing
        "buyers":   "#A32D2D",   # red — buyers have power
    }
    market_labels = {
        "sellers":  "Seller's market",
        "balanced": "Balanced market",
        "buyers":   "Buyer's market",
    }

    for _, row in df.iterrows():
        snlr   = row.get("snlr", 0.5)
        if snlr > 0.6:
            mtype = "sellers"
        elif snlr < 0.4:
            mtype = "buyers"
        else:
            mtype = "balanced"

        color = market_colors[mtype]

        popup_html = f"""
        <div style="font-family:Arial;min-width:200px">
          <h4 style="margin:0 0 8px">{row['city']}</h4>
          <p style="font-size:13px;margin:0">
            <b>{market_labels[mtype]}</b><br>
            SNLR: {snlr:.2f}<br>
            Active listings: {row.get('active_listings', 0):,.0f}<br>
            Days on market: {row.get('days_on_market', 0):.0f}
          </p>
        </div>
        """
        tooltip = f"{row['city']}: {market_labels[mtype]} (SNLR={snlr:.2f})"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=14,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=tooltip,
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-family:Arial;font-size:13px">
      <b>Market Type (SNLR)</b><br>
      <span style="color:#27500A">&#9679;</span> Seller's market (&gt;0.6)<br>
      <span style="color:#BA7517">&#9679;</span> Balanced (0.4–0.6)<br>
      <span style="color:#A32D2D">&#9679;</span> Buyer's market (&lt;0.4)<br>
      <i style="font-size:11px;color:#888">SNLR = Sales / New Listings</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = cfg.maps_dir / "04_market_type_map.html"
    m.save(str(path))
    logger.info(f"Saved: {path}")


# ── Map 5: Combined dashboard map ────────────────────────────────────────────

def map_combined_plotly(df: pd.DataFrame):
    """
    Combined map using Plotly — reliable layer switching via dropdown.
    Fallback for Folium FeatureGroup compatibility issues.
    """
    import plotly.graph_objects as go

    layers = {
        "Average Price":       ("avg_price",          "Blues",  "$,.0f"),
        "Affordability Ratio": ("affordability_ratio", "RdYlGn_r", ".1f"),
        "YoY Price Change":    ("price_mom_12m",       "RdYlGn",  "+.1f"),
    }

    fig = go.Figure()

    for i, (layer_name, (col, colorscale, fmt)) in enumerate(layers.items()):
        vals = df[col].fillna(0)
        fig.add_trace(go.Scattermapbox(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
            marker=dict(
                size=16,
                color=vals,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=layer_name, x=1.0),
            ),
            text=[
                f"<b>{row['city']}</b><br>"
                f"Avg Price: ${row['avg_price']:,.0f}<br>"
                f"YoY: {row.get('price_mom_12m',0):+.1f}%<br>"
                f"Affordability: {row.get('affordability_ratio',0):.1f}x<br>"
                f"Days on Market: {row.get('days_on_market',0):.0f}"
                for _, row in df.iterrows()
            ],
            hoverinfo="text",
            name=layer_name,
            visible=(i == 0),
        ))

    # Dropdown buttons
    buttons = []
    for i, layer_name in enumerate(layers.keys()):
        visibility = [j == i for j in range(len(layers))]
        buttons.append(dict(
            label=layer_name,
            method="update",
            args=[{"visible": visibility}],
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=56.0, lon=-96.0),
            zoom=3.2,
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            buttons=buttons,
            bgcolor="white",
            bordercolor="#ccc",
            font=dict(size=13),
        )],
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title="Canadian Housing Market Intelligence — Select a metric",
    )

    path = cfg.maps_dir / "05_combined_map.html"
    fig.write_html(str(path), include_plotlyjs=True)
    logger.info(f"Saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg.maps_dir.mkdir(parents=True, exist_ok=True)
    df    = load_latest()
    master = load_master()

    logger.info("Generating maps...")
    map_avg_price(df)
    map_affordability(df)
    map_yoy_change(df)
    map_market_type(df)
    map_combined_plotly(df)

    logger.info("=" * 60)
    logger.info("Milestone 3 complete. Open maps in your browser:")
    logger.info("  reports/maps/05_combined_map.html   <- START HERE")
    logger.info("  reports/maps/01_avg_price_map.html")
    logger.info("  reports/maps/02_affordability_map.html")
    logger.info("Next: python -m src.models.forecast")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
