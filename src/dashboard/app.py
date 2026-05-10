"""
src/dashboard/app.py

Streamlit dashboard for Canadian Real Estate Market Intelligence.

Usage (from project root):
    streamlit run src/dashboard/app.py

Sections:
    1. National overview — KPI cards + combined map
    2. City deep-dive  — price history + forecast + affordability
    3. Market analysis — SNLR heatmap + rate hike story
    4. Model comparison — SARIMA vs XGBoost MAPE table
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Auto-build data if running on Streamlit Cloud
if not (Path("data/processed/master_dataset.parquet")).exists():
    from src.collection.build_dataset import build_master_dataset
    from src.models.forecast import main as run_forecast
    build_master_dataset()
    run_forecast()

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Canadian Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
CHARTS_DIR    = Path("reports/charts")
MAPS_DIR      = Path("reports/maps")

# ── Data loading (cached so it only runs once) ────────────────────────────────

@st.cache_data
def load_master():
    df = pd.read_parquet(PROCESSED_DIR / "master_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_latest():
    df = pd.read_parquet(PROCESSED_DIR / "city_latest.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_forecasts():
    path = PROCESSED_DIR / "forecasts.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #185FA5;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
        margin: 0;
        font-family: Arial;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #1a1a1a;
        margin: 4px 0 0;
        font-family: Arial;
    }
    .metric-delta {
        font-size: 13px;
        margin: 2px 0 0;
        font-family: Arial;
    }
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #185FA5;
        border-bottom: 2px solid #E6F1FB;
        padding-bottom: 8px;
        margin: 24px 0 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(master: pd.DataFrame) -> str:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/d/d9/Flag_of_Canada_%28Pantone%29.svg",
        width=80,
    )
    st.sidebar.title("🏠 Canadian Real Estate")
    st.sidebar.markdown("*Market Intelligence Dashboard*")
    st.sidebar.divider()

    cities = sorted(master["city"].unique().tolist())
    selected_city = st.sidebar.selectbox(
        "Select a city", cities, index=cities.index("Toronto")
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Data sources**")
    st.sidebar.markdown("- CREA MLS Statistics")
    st.sidebar.markdown("- Bank of Canada rates")
    st.sidebar.markdown("- Statistics Canada")
    st.sidebar.divider()
    st.sidebar.markdown("**Models**")
    st.sidebar.markdown("- SARIMA (seasonal baseline)")
    st.sidebar.markdown("- XGBoost (lag features)")
    st.sidebar.divider()
    st.sidebar.caption("Built by Priyank | 2025")

    return selected_city


# ── Section 1: National KPI cards ────────────────────────────────────────────

def render_national_kpis(latest: pd.DataFrame):
    st.markdown('<p class="section-header">National Overview</p>',
                unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [
        ("Cities tracked",     f"{latest['city'].nunique()}",           None),
        ("Highest avg price",  f"${latest['avg_price'].max():,.0f}",
         latest.loc[latest['avg_price'].idxmax(), 'city']),
        ("Most affordable",
         latest.loc[latest['affordability_ratio'].idxmin(), 'city'],
         f"{latest['affordability_ratio'].min():.1f}x income"),
        ("Least affordable",
         latest.loc[latest['affordability_ratio'].idxmax(), 'city'],
         f"{latest['affordability_ratio'].max():.1f}x income"),
        ("Avg days on market",
         f"{latest['days_on_market'].mean():.0f} days",             None),
    ]

    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            delta_html = (
                f'<p class="metric-delta" style="color:#666">{delta}</p>'
                if delta else ""
            )
            col.markdown(f"""
            <div class="metric-card">
              <p class="metric-label">{label}</p>
              <p class="metric-value">{value}</p>
              {delta_html}
            </div>
            """, unsafe_allow_html=True)


# ── Section 2: National map ───────────────────────────────────────────────────

def render_national_map(latest: pd.DataFrame):
    st.markdown('<p class="section-header">Interactive Map</p>',
                unsafe_allow_html=True)

    # Check if combined Plotly map exists
    map_path = MAPS_DIR / "05_combined_map.html"
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()
        components.html(map_html, height=520, scrolling=False)
    else:
        st.info("Map not found. Run: python -m src.visualization.maps")


# ── Section 3: City deep-dive ─────────────────────────────────────────────────

def render_city_overview(master: pd.DataFrame,
                         latest: pd.DataFrame,
                         city: str):
    st.markdown(f'<p class="section-header">{city} — City Deep Dive</p>',
                unsafe_allow_html=True)

    city_data   = master[master["city"] == city].sort_values("date")
    city_latest = latest[latest["city"] == city].iloc[0]

    # KPI row for selected city
    yoy   = city_latest.get("price_mom_12m", 0)
    ratio = city_latest.get("affordability_ratio", 0)
    snlr  = city_latest.get("snlr", 0.5)

    if snlr > 0.6:
        market_type = "Seller's market 🟢"
    elif snlr < 0.4:
        market_type = "Buyer's market 🔴"
    else:
        market_type = "Balanced market 🟡"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Price",    f"${city_latest['avg_price']:,.0f}",
              f"{yoy:+.1f}% YoY")
    c2.metric("Median Price",     f"${city_latest['median_price']:,.0f}")
    c3.metric("Affordability",    f"{ratio:.1f}x income",
              "vs 5x global threshold")
    c4.metric("Market Type",      market_type,
              f"SNLR: {snlr:.2f}")

    st.divider()

    # Price history chart
    col_left, col_right = st.columns([2, 1])

    with col_left:
        fig = go.Figure()

        # Price line
        fig.add_trace(go.Scatter(
            x=city_data["date"],
            y=city_data["avg_price"],
            name="Avg Price",
            line=dict(color="#185FA5", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(24,95,165,0.07)",
        ))

        # BoC rate shading for hiking period
        fig.add_vrect(
            x0="2022-03-01", x1="2023-07-01",
            fillcolor="rgba(226,75,74,0.07)",
            layer="below", line_width=0,
            annotation_text="Rate hiking cycle",
            annotation_font_size=10,
            annotation_font_color="#A32D2D",
        )

        fig.update_layout(
            title=f"{city}: Average Price History",
            yaxis_tickformat="$,.0f",
            template="plotly_white",
            height=340,
            margin=dict(t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Affordability over time
        fig2 = go.Figure(go.Scatter(
            x=city_data["date"],
            y=city_data["affordability_ratio"],
            line=dict(color="#E24B4A", width=2),
            fill="tozeroy",
            fillcolor="rgba(226,75,74,0.07)",
        ))
        fig2.add_hline(y=5, line_dash="dash", line_color="#3B6D11",
                       annotation_text="5x threshold",
                       annotation_font_size=10)
        fig2.update_layout(
            title="Affordability Ratio (x income)",
            template="plotly_white",
            height=340,
            margin=dict(t=40, b=20),
            yaxis_title="Price / Annual Income",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Section 4: Forecast ───────────────────────────────────────────────────────

def render_forecast(master: pd.DataFrame,
                    forecasts: pd.DataFrame,
                    city: str):
    st.markdown('<p class="section-header">12-Month Price Forecast</p>',
                unsafe_allow_html=True)

    if forecasts is None:
        st.info("Forecasts not found. Run: python -m src.models.forecast")
        return

    city_fc = forecasts[forecasts["city"] == city]
    if city_fc.empty:
        st.warning(f"No forecast available for {city}")
        return

    city_hist = master[master["city"] == city].sort_values("date")

    # Model selector
    available_models = city_fc["model"].unique().tolist()
    selected_model   = st.selectbox(
        "Select forecast model",
        available_models,
        index=0,
        key=f"model_{city}",
    )

    model_fc  = city_fc[city_fc["model"] == selected_model]
    future_fc = model_fc[model_fc["is_future"] == True]
    test_fc   = model_fc[model_fc["is_future"] == False]
    test_mape = model_fc["test_mape"].iloc[0]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=city_hist["date"], y=city_hist["avg_price"],
        name="Historical",
        line=dict(color="#185FA5", width=2),
    ))

    # Test period forecast
    if not test_fc.empty:
        fig.add_trace(go.Scatter(
            x=test_fc["date"], y=test_fc["forecast"],
            name=f"{selected_model} (test, MAPE={test_mape:.1f}%)",
            line=dict(color="#BA7517", width=1.5, dash="dash"),
        ))

    # Future forecast + uncertainty band
    if not future_fc.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([future_fc["date"], future_fc["date"].iloc[::-1]]),
            y=pd.concat([future_fc["upper"], future_fc["lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(226,75,74,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% confidence interval",
        ))
        fig.add_trace(go.Scatter(
            x=future_fc["date"], y=future_fc["forecast"],
            name=f"{selected_model} forecast",
            line=dict(color="#E24B4A", width=2.5),
        ))

        # Annotation: 12-month forecast end price
        end_price = future_fc["forecast"].iloc[-1]
        current   = city_hist["avg_price"].iloc[-1]
        chg       = (end_price - current) / current * 100
        st.info(
            f"**{selected_model} 12-month forecast:** "
            f"${end_price:,.0f} "
            f"({'↑' if chg > 0 else '↓'}{abs(chg):.1f}% from current ${current:,.0f})"
        )

    fig.update_layout(
        title=f"{city} — {selected_model} Price Forecast",
        yaxis_tickformat="$,.0f",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Section 5: Rate hike story ────────────────────────────────────────────────

def render_rate_hike_story(master: pd.DataFrame):
    st.markdown('<p class="section-header">The Rate Hike Story</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Between March 2022 and July 2023, the Bank of Canada raised its "
        "overnight rate from **0.25% to 5.00%** — the fastest hiking cycle "
        "in Canadian history. The impact on housing prices was immediate and severe."
    )

    toronto = master[master["city"] == "Toronto"].sort_values("date")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=toronto["date"], y=toronto["boc_rate"],
        name="BoC Rate (%)",
        line=dict(color="#E24B4A", width=2.5),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=toronto["date"], y=toronto["avg_price"],
        name="Toronto Avg Price",
        line=dict(color="#185FA5", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(24,95,165,0.07)",
    ), secondary_y=True)

    fig.add_vrect(
        x0="2022-03-01", x1="2023-07-01",
        fillcolor="rgba(226,75,74,0.08)",
        layer="below", line_width=0,
        annotation_text="Hiking cycle (+475bps)",
        annotation_font_color="#A32D2D",
        annotation_font_size=11,
    )

    fig.update_layout(
        template="plotly_white", height=380,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=20),
    )
    fig.update_yaxes(title_text="BoC Rate (%)", secondary_y=False,
                     ticksuffix="%", range=[0, 6])
    fig.update_yaxes(title_text="Toronto Avg Price (CAD)",
                     secondary_y=True, tickformat="$,.0f")

    st.plotly_chart(fig, use_container_width=True)

    # Correlation stat
    corr = toronto[["boc_rate", "avg_price"]].corr().iloc[0, 1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Rate-Price Correlation", f"{corr:.2f}",
                "Negative = inverse relationship")
    col2.metric("Rate increase", "+475 bps",
                "0.25% → 5.00% (Mar 2022–Jul 2023)")
    col3.metric("Toronto peak-to-trough", "~-16%",
                "During the hiking cycle")


# ── Section 6: Model comparison ───────────────────────────────────────────────

def render_model_comparison(forecasts: pd.DataFrame):
    st.markdown('<p class="section-header">Model Performance Comparison</p>',
                unsafe_allow_html=True)

    if forecasts is None:
        st.info("Run: python -m src.models.forecast")
        return

    # MAPE summary table
    mape_df = (
        forecasts[["city", "model", "test_mape"]]
        .drop_duplicates()
        .pivot(index="city", columns="model", values="test_mape")
        .round(2)
    )
    mape_df["Best Model"] = mape_df.idxmin(axis=1)
    mape_df.columns.name = None

    st.dataframe(
        mape_df.style
        .format("{:.1f}%", subset=[c for c in mape_df.columns
                                    if c != "Best Model"])
        .background_gradient(cmap="RdYlGn_r",
                              subset=[c for c in mape_df.columns
                                       if c != "Best Model"])
        .set_caption("MAPE (%) — lower is better. Green = accurate, Red = less accurate"),
        use_container_width=True,
    )

    # Bar chart
    records = []
    for _, row in mape_df.iterrows():
        for model in [c for c in mape_df.columns if c != "Best Model"]:
            if not pd.isna(row.get(model)):
                records.append({"City": row.name, "Model": model,
                                 "MAPE": row[model]})
    if records:
        fig = px.bar(
            pd.DataFrame(records),
            x="City", y="MAPE", color="Model", barmode="group",
            color_discrete_map={"SARIMA": "#3B6D11", "XGBoost": "#BA7517"},
            template="plotly_white", height=360,
            labels={"MAPE": "MAPE (%)"},
            title="Forecast Accuracy by City and Model",
            text="MAPE",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(y=8, line_dash="dash", line_color="#888",
                      annotation_text="8% target")
        st.plotly_chart(fig, use_container_width=True)


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    # Load data
    try:
        master    = load_master()
        latest    = load_latest()
        forecasts = load_forecasts()
    except FileNotFoundError as e:
        st.error(f"Data not found: {e}")
        st.info("Run these commands first:\n"
                "1. python -m src.collection.build_dataset\n"
                "2. python -m src.models.forecast")
        return

    # Header
    st.title("🏠 Canadian Real Estate Market Intelligence")
    st.markdown(
        "Interactive analysis of housing prices, affordability, and 12-month "
        "forecasts across 10 major Canadian cities. "
        "Data: CREA MLS Statistics + Bank of Canada rates."
    )
    st.divider()

    # Sidebar — city selector
    selected_city = render_sidebar(master)

    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ National Overview",
        "🏙️ City Deep Dive",
        "📈 Rate Hike Story",
        "🤖 Model Comparison",
    ])

    with tab1:
        render_national_kpis(latest)
        render_national_map(latest)

    with tab2:
        render_city_overview(master, latest, selected_city)
        render_forecast(master, forecasts, selected_city)

    with tab3:
        render_rate_hike_story(master)

    with tab4:
        render_model_comparison(forecasts)


if __name__ == "__main__":
    main()
