"""
Volume Screener : current volume > average volume  (Forex + Commodities)
------------------------------------------------------------------------
Install : pip install yfinance pandas numpy
          pip install MetaTrader5      (optional, Windows only, for spot forex)

Run     : python volume_screener.py                      # 1h bars, yfinance
          python volume_screener.py --tf 1d              # daily bars
          python volume_screener.py --min-rvol 1.5       # only strong spikes
          python volume_screener.py --cross-only         # only fresh crosses
          python volume_screener.py --mt5 --suffix ""    # spot forex tick volume

DATA NOTE
 * Yahoo spot forex (EURUSD=X) has ZERO volume, so here forex is tracked via
   CME currency futures (6E, 6B, 6J ...) which have real volume.
 * For real spot-pair activity use --mt5 (broker tick volume).
 * Commodities = futures (GC, CL, ...) with real volume.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

FOREX_FUT = {  # CME currency futures (proxy for spot pairs)
    "6E=F": "EURUSD", "6B=F": "GBPUSD", "6J=F": "USDJPY(inv)",
    "6A=F": "AUDUSD", "6C=F": "USDCAD(inv)", "6S=F": "USDCHF(inv)",
    "6N=F": "NZDUSD",
}
COMMODITIES = {
    "GC=F": "Gold", "SI=F": "Silver", "HG=F": "Copper", "PL=F": "Platinum",
    "CL=F": "WTI Crude", "BZ=F": "Brent", "NG=F": "Natural Gas",
    "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans",
    "KC=F": "Coffee", "SB=F": "Sugar", "CT=F": "Cotton",
}
MT5_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
             "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD",
             "USOIL"]


# ------------------------------------------------------------------ data
def load_yf(sym, tf):
    interval, period = ("1h", "60d") if tf == "1h" else ("1d", "1y")
    df = yf.download(sym, interval=interval, period=period,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def load_mt5(sym, tf, mt5):
    frame = mt5.TIMEFRAME_H1 if tf == "1h" else mt5.TIMEFRAME_D1
    rates = mt5.copy_rates_from_pos(sym, frame, 0, 400)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ------------------------------------------------------------ indicators
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# --------------------------------------------------------------- analyse
def analyse(name, df, n, live):
    if df is None or len(df) < n + 60:
        return None
    if not live:                      # drop the still-forming bar
        df = df.iloc[:-1]
    vol = df["Volume"]
    if vol.tail(n).sum() == 0:
        return None                   # no volume data for this symbol

    avg = vol.rolling(n).mean().shift(1)     # avg of previous n bars
    cur, prev = vol.iloc[-1], vol.iloc[-2]
    rvol = cur / avg.iloc[-1]
    crossed = cur > avg.iloc[-1] and prev <= avg.iloc[-2]

    c = df["Close"]
    e20, e50, r, a = ema(c, 20), ema(c, 50), rsi(c).iloc[-1], atr(df).iloc[-1]
    last = df.iloc[-1]
    bull_bar = last.Close > last.Open

    # technical bias (same idea as your stock strategy)
    if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1] and bull_bar and r > 50:
        bias = "BUY"
    elif c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1] and not bull_bar and r < 50:
        bias = "SELL"
    else:
        bias = "WAIT"

    px = c.iloc[-1]
    sl = px - 1.5 * a if bias == "BUY" else px + 1.5 * a if bias == "SELL" else np.nan
    dec = 2 if px > 50 else 5
    return {
        "Symbol": name,
        "RVOL": round(rvol, 2),
        "Cross": "YES" if crossed else "",
        "Volume": int(cur),
        "AvgVol": int(avg.iloc[-1]),
        "Bias": bias,
        "Price": round(px, dec),
        "SL(1.5ATR)": round(sl, dec) if not np.isnan(sl) else "",
        "RSI": round(r, 1),
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["1h", "1d"], default="1h")
    ap.add_argument("--avg", type=int, default=20, help="avg volume period")
    ap.add_argument("--min-rvol", type=float, default=1.0)
    ap.add_argument("--cross-only", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="use the still-forming bar (volume is partial)")
    ap.add_argument("--mt5", action="store_true")
    ap.add_argument("--suffix", default="", help="broker symbol suffix, e.g. m or .a")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    if args.mt5:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise SystemExit("MT5 initialize failed: open terminal & login first")
        for s in MT5_PAIRS:
            rows.append(analyse(s, load_mt5(s + args.suffix, args.tf, mt5),
                                args.avg, args.live))
        mt5.shutdown()
    else:
        for tick, nm in {**FOREX_FUT, **COMMODITIES}.items():
            rows.append(analyse(nm, load_yf(tick, args.tf), args.avg, args.live))

    rows = [r for r in rows if r]
    if not rows:
        print("No data returned.")
        return
    out = pd.DataFrame(rows).sort_values("RVOL", ascending=False)
    out = out[out["RVOL"] >= args.min_rvol]
    if args.cross_only:
        out = out[out["Cross"] == "YES"]

    pd.set_option("display.width", 200)
    print(f"\nVolume screener | tf={args.tf} | avg={args.avg} bars | RVOL >= {args.min_rvol}\n")
    print(out.to_string(index=False) if len(out) else "Kahi pan avg volume cross kelela nahi.")
    if args.csv:
        out.to_csv(args.csv, index=False)
        print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()
