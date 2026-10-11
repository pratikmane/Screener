"""
Forex Screener  (D1 trend + H4 structure + currency strength)
-------------------------------------------------------------
Install :  pip install yfinance pandas numpy
Run     :  python forex_screener.py
           python forex_screener.py --min-score 60 --csv out.csv

NOTE: Yahoo forex volume = 0, so volume is NOT used. Instead we use
volatility (ATR), trend strength (ADX) and currency strength.
Screener = shortlist tool. Final entry decision = your own chart analysis.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURCHF", "GBPAUD",
    "GBPCAD", "AUDNZD", "CADJPY", "NZDJPY", "CHFJPY", "EURCAD", "AUDCAD",
]


# ------------------------------------------------------------------ data
def fetch(pair, interval, period):
    df = yf.download(f"{pair}=X", interval=interval, period=period,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def to_h4(df1h):
    return (df1h.resample("4h")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna())


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
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------------------ currency strength
def currency_strength(daily, lookback=5):
    """Avg % move of each currency vs all others over `lookback` days."""
    score, count = {}, {}
    for pair, d in daily.items():
        ret = d["Close"].iloc[-1] / d["Close"].iloc[-1 - lookback] - 1
        b, q = pair[:3], pair[3:]
        score[b] = score.get(b, 0) + ret
        score[q] = score.get(q, 0) - ret
        count[b] = count.get(b, 0) + 1
        count[q] = count.get(q, 0) + 1
    return {c: 100 * score[c] / count[c] for c in score}


# ---------------------------------------------------------------- scoring
def analyse(pair, d1, h4, strength):
    d1 = d1.copy()
    h4 = h4.copy()
    if len(d1) < 210 or len(h4) < 60:
        return None

    d1["e50"], d1["e200"] = ema(d1["Close"], 50), ema(d1["Close"], 200)
    h4["e20"], h4["e50"] = ema(h4["Close"], 20), ema(h4["Close"], 50)
    h4["atr"], h4["adx"], h4["rsi"] = atr(h4), adx(h4), rsi(h4["Close"])
    h4["hh"] = h4["High"].rolling(20).max().shift()
    h4["ll"] = h4["Low"].rolling(20).min().shift()

    D, H = d1.iloc[-1], h4.iloc[-1]

    # direction from D1 trend
    if D.Close > D.e50 > D.e200:
        side = 1
    elif D.Close < D.e50 < D.e200:
        side = -1
    else:
        return None  # no clear higher-timeframe trend -> skip

    score, tags = 25, ["D1 trend"]

    # H4 EMA alignment
    if side == 1 and H.e20 > H.e50:
        score += 20; tags.append("H4 EMA aligned")
    elif side == -1 and H.e20 < H.e50:
        score += 20; tags.append("H4 EMA aligned")

    # trend strength
    if H.adx >= 22:
        score += 15; tags.append(f"ADX {H.adx:.0f}")

    # setup type
    last3 = h4.iloc[-3:]
    near_ema = ((last3["Low"] <= last3["e20"] + 0.5 * last3["atr"]) &
                (last3["High"] >= last3["e20"] - 0.5 * last3["atr"])).any() or \
               ((last3["Low"] <= last3["e50"] + 0.5 * last3["atr"]) &
                (last3["High"] >= last3["e50"] - 0.5 * last3["atr"])).any()
    setup = "Trend"
    if side == 1:
        if near_ema and 38 <= H.rsi <= 58 and H.Close > H.e50:
            score += 20; setup = "Pullback"; tags.append("pullback to EMA")
        elif H.Close > H.hh:
            score += 15; setup = "Breakout"; tags.append("20-bar breakout")
    else:
        if near_ema and 42 <= H.rsi <= 62 and H.Close < H.e50:
            score += 20; setup = "Pullback"; tags.append("pullback to EMA")
        elif H.Close < H.ll:
            score += 15; setup = "Breakout"; tags.append("20-bar breakdown")

    # currency strength differential
    b, q = pair[:3], pair[3:]
    diff = strength[b] - strength[q]
    if diff * side > 0:
        score += 10; tags.append(f"CCY strength {diff:+.2f}")
    if abs(diff) > 0.5 and diff * side < 0:
        score -= 10; tags.append("CCY strength against")

    # healthy volatility (current ATR vs 50-bar average ATR)
    ratio = H.atr / h4["atr"].rolling(50).mean().iloc[-1]
    if 0.8 <= ratio <= 1.8:
        score += 10
    tags.append(f"ATRx{ratio:.2f}")

    pip = 0.01 if pair.endswith("JPY") else 0.0001
    risk = 1.5 * H.atr
    entry = H.Close
    return {
        "Pair": pair,
        "Side": "BUY" if side == 1 else "SELL",
        "Score": int(score),
        "Setup": setup,
        "Price": round(entry, 3 if pip == 0.01 else 5),
        "SL": round(entry - side * risk, 3 if pip == 0.01 else 5),
        "TP 2R": round(entry + side * 2 * risk, 3 if pip == 0.01 else 5),
        "SL pips": round(risk / pip, 1),
        "RSI": round(H.rsi, 1),
        "Notes": ", ".join(tags),
    }


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=int, default=50)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    daily, h4s = {}, {}
    for p in PAIRS:
        d1 = fetch(p, "1d", "2y")
        h1 = fetch(p, "1h", "60d")
        if d1 is None or h1 is None:
            print(f"skip {p}: no data")
            continue
        daily[p], h4s[p] = d1, to_h4(h1)

    strength = currency_strength(daily)
    print("\nCurrency strength (5-day):")
    for c, v in sorted(strength.items(), key=lambda x: -x[1]):
        print(f"  {c}: {v:+.2f}")

    rows = [r for p in daily
            if (r := analyse(p, daily[p], h4s[p], strength))]
    if not rows:
        print("\nNo setups found.")
        return

    out = (pd.DataFrame(rows).sort_values("Score", ascending=False)
           .query("Score >= @args.min_score").reset_index(drop=True))
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(f"\nShortlist (score >= {args.min_score}):\n")
    print(out.to_string(index=False) if len(out) else "Nothing above min score.")
    if args.csv:
        out.to_csv(args.csv, index=False)
        print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()
