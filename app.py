"""
Mobile Market Screener (Streamlit)
Tab 1: Volume > Avg Volume  (currency futures + commodities)
Tab 2: Forex trend screener (D1 + H4 + currency strength)
"""
import pandas as pd
import streamlit as st

import forex_screener as fx
import volume_screener as vs

st.set_page_config(page_title="Screener", page_icon="📈", layout="centered")
st.title("📈 Market Screener")

ICON = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "WAIT": "⚪ WAIT"}


# ----------------------------------------------------------- cached scans
@st.cache_data(ttl=600, show_spinner=False)
def scan_volume(tf: str, n: int, live: bool) -> pd.DataFrame:
    rows = []
    for tick, name in {**vs.FOREX_FUT, **vs.COMMODITIES}.items():
        try:
            rows.append(vs.analyse(name, vs.load_yf(tick, tf), n, live))
        except Exception:
            pass
    rows = [r for r in rows if r]
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def scan_trend():
    daily, h4s = {}, {}
    for p in fx.PAIRS:
        try:
            d1, h1 = fx.fetch(p, "1d", "2y"), fx.fetch(p, "1h", "60d")
            if d1 is not None and h1 is not None:
                daily[p], h4s[p] = d1, fx.to_h4(h1)
        except Exception:
            pass
    strength = fx.currency_strength(daily)
    rows = [r for p in daily if (r := fx.analyse(p, daily[p], h4s[p], strength))]
    return strength, pd.DataFrame(rows)


# -------------------------------------------------------------------- UI
if st.button("🔄 Refresh data", use_container_width=True):
    st.cache_data.clear()

tab1, tab2 = st.tabs(["📊 Volume", "📈 Trend"])

with tab1:
    c1, c2 = st.columns(2)
    tf = c1.radio("Timeframe", ["1h", "1d"], horizontal=True)
    n = c2.number_input("Avg period", 5, 100, 20)
    min_rvol = st.slider("Min RVOL (volume ÷ avg)", 1.0, 4.0, 1.0, 0.1)
    cross_only = st.checkbox("Only fresh crosses")
    live = st.checkbox("Include running bar (volume partial)")

    with st.spinner("Scanning..."):
        df = scan_volume(tf, int(n), live)

    if df.empty:
        st.warning("Data aali nahi. Thodya velane Refresh kar.")
    else:
        df = df[df["RVOL"] >= min_rvol].sort_values("RVOL", ascending=False)
        if cross_only:
            df = df[df["Cross"] == "YES"]
        if df.empty:
            st.info("Kahi pan avg volume cross kelela nahi.")
        else:
            df["Bias"] = df["Bias"].map(ICON)
            st.caption(f"{len(df)} symbols | sorted by RVOL")
            st.dataframe(
                df[["Symbol", "RVOL", "Cross", "Bias", "Price", "SL(1.5ATR)", "RSI"]],
                hide_index=True, use_container_width=True)

with tab2:
    min_score = st.slider("Min score", 0, 100, 50, 5)
    with st.spinner("Scanning pairs (30-60 sec)..."):
        strength, tdf = scan_trend()

    st.subheader("Currency strength (5-day)")
    st.bar_chart(pd.Series(strength).sort_values())

    st.subheader("Shortlist")
    if tdf.empty:
        st.info("Kahi setup nahi.")
    else:
        tdf = tdf[tdf["Score"] >= min_score].sort_values("Score", ascending=False)
        if tdf.empty:
            st.info("Ya score var kahi nahi. Score kami kar.")
        else:
            tdf["Side"] = tdf["Side"].map(ICON)
            st.dataframe(
                tdf[["Pair", "Side", "Score", "Setup", "Price", "SL", "TP 2R", "SL pips"]],
                hide_index=True, use_container_width=True)
            with st.expander("Details"):
                st.dataframe(tdf[["Pair", "RSI", "Notes"]],
                             hide_index=True, use_container_width=True)

st.caption("Shortlist tool aahe, trade advice nahi. Final decision tuzya chart analysis var.")
