"""
Backtest the strategy on ~50 days of 15m Twelve Data candles.
Env: TWELVE_DATA_KEY (secret), SYMBOLS (default "EURUSD")
Spreads below are ASSUMED typical values - edit to match your XM account.
"""
import os

import pandas as pd
import requests

from strategy import backtest

KEY = os.environ["TWELVE_DATA_KEY"]
SYMBOLS = [s.strip() for s in os.environ.get("SYMBOLS", "EURUSD").split(",") if s.strip()]
LOTS, SL_USD, TP_USD = 0.01, 2.0, 4.0
CONTRACT = {"EURUSD": 100000, "GBPUSD": 100000, "AUDUSD": 100000, "NZDUSD": 100000,
            "XAUUSD": 100, "XAGUSD": 5000}
SPREAD = {"EURUSD": 0.00012, "GBPUSD": 0.00015, "AUDUSD": 0.00015, "NZDUSD": 0.00018,
          "XAUUSD": 0.25, "XAGUSD": 0.03}

for s in SYMBOLS:
    if s not in CONTRACT:
        print(s, "- only USD-quoted symbols supported:", ", ".join(CONTRACT)); continue
    td_sym = s[:3] + "/" + s[3:]
    js = requests.get("https://api.twelvedata.com/time_series",
                      params={"symbol": td_sym, "interval": "15min", "outputsize": 5000,
                              "apikey": KEY}, timeout=60).json()
    if "values" not in js:
        print(s, "- data error:", js.get("message")); continue
    df = pd.DataFrame(js["values"]).iloc[::-1].reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c])
    dist_sl = SL_USD / (LOTS * CONTRACT[s])
    dist_tp = TP_USD / (LOTS * CONTRACT[s])
    res = backtest(df, dist_sl, dist_tp, spread=SPREAD[s], sl_usd=SL_USD, tp_usd=TP_USD)
    print(f"\n{s}: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]} ({len(df)} candles)")
    print(f"  SL {dist_sl:g} / TP {dist_tp:g} price units | assumed spread {SPREAD[s]:g}")
    for k, v in res.items():
        print(f"  {k}: {v}")
