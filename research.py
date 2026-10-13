"""
Strategy research: tests several classic, logically different 1H strategies on ~14 months of
Twelve Data candles, with ATR-based stops, spread costs and an in-sample / out-of-sample split.
Results are in R (1R = the stop distance), so they work for any symbol.

Env: TWELVE_DATA_KEY (secret), SYMBOLS (default EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD)
Spreads are ASSUMED typical values (price units) - edit to match your broker.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

from strategy import structure

KEY = os.environ.get("TWELVE_DATA_KEY", "")
SYMBOLS = [s.strip() for s in os.environ.get("SYMBOLS", "EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD").split(",") if s.strip()]
SPREAD = {"EURUSD": 0.00012, "GBPUSD": 0.00015, "AUDUSD": 0.00015, "NZDUSD": 0.00018,
          "USDJPY": 0.012, "USDCAD": 0.00018, "USDCHF": 0.00018, "XAUUSD": 0.25, "XAGUSD": 0.03}
PAGES = 2            # 5000 hourly candles per page (~7 months each)
MAX_HOLD = 48        # close a trade after this many bars if neither SL nor TP was hit
IS_FRAC = 0.6        # first 60% = in-sample, last 40% = out-of-sample


# ----------------------------------------------------------------- data
def fetch(sym):
    frames, end = [], None
    for _ in range(PAGES):
        p = {"symbol": sym[:3] + "/" + sym[3:], "interval": "1h", "outputsize": 5000,
             "timezone": "UTC", "apikey": KEY}
        if end:
            p["end_date"] = end
        js = requests.get("https://api.twelvedata.com/time_series", params=p, timeout=90).json()
        time.sleep(8)                                    # free plan: 8 requests / minute
        if "values" not in js:
            print(f"  {sym}: {js.get('message', 'no data')}")
            break
        d = pd.DataFrame(js["values"])
        frames.append(d)
        end = d["datetime"].iloc[-1]                     # oldest candle of this page
        if len(d) < 4000:
            break
    if not frames:
        return None
    df = pd.concat(frames).drop_duplicates("datetime")
    df["time"] = pd.to_datetime(df["datetime"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c])
    return df.sort_values("time").reset_index(drop=True)[["time", "open", "high", "low", "close"]]


# ----------------------------------------------------------- indicators
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def wilder(s, n): return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(c, n):
    d = c.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def prep(df):
    d = df.copy()
    pc = d["close"].shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = wilder(tr, 14)
    d["e20"], d["e50"], d["e200"] = ema(d.close, 20), ema(d.close, 50), ema(d.close, 200)
    d["rsi3"] = rsi(d.close, 3)
    d["hh20"], d["ll20"] = d.high.rolling(20).max().shift(), d.low.rolling(20).min().shift()
    d["hh55"], d["ll55"] = d.high.rolling(55).max().shift(), d.low.rolling(55).min().shift()
    d["hour"], d["day"] = d.time.dt.hour, d.time.dt.date
    return d


# ----------------------------------------------------------- strategies
# each returns an int array: +1 buy / -1 sell / 0 none, decided on the CLOSE of bar i
def s_trend_pullback(d):
    up = (d.e50 > d.e200) & (d.close > d.e200)
    dn = (d.e50 < d.e200) & (d.close < d.e200)
    long = up & (d.low <= d.e20) & (d.close > d.e20) & (d.close > d.open)
    short = dn & (d.high >= d.e20) & (d.close < d.e20) & (d.close < d.open)
    return (long.astype(int) - short.astype(int)).to_numpy()


def s_donchian(n):
    def f(d):
        up, dn = d.e50 > d.e200, d.e50 < d.e200
        long = (d.close > d[f"hh{n}"]) & up
        short = (d.close < d[f"ll{n}"]) & dn
        return (long.astype(int) - short.astype(int)).to_numpy()
    return f


def s_london_breakout(d):
    asia = d[d.hour < 7].groupby("day").agg(ah=("high", "max"), al=("low", "min"))
    m = d.join(asia, on="day")
    win = (m.hour >= 7) & (m.hour <= 11)
    raw = np.where(win & (m.close > m.ah), 1, np.where(win & (m.close < m.al), -1, 0))
    out, last_day = np.zeros(len(d), dtype=int), None
    for i, (dy, s) in enumerate(zip(m.day, raw)):        # first breakout of each day only
        if s != 0 and dy != last_day:
            out[i], last_day = s, dy
    return out


def s_rsi_meanrev(d):
    long = (d.rsi3 < 15) & (d.close > d.e200)
    short = (d.rsi3 > 85) & (d.close < d.e200)
    return (long.astype(int) - short.astype(int)).to_numpy()


def _bos(d, counter):
    st, bos = structure(d.rename(columns=str.lower))
    O, C = d.open.to_numpy(), d.close.to_numpy()
    out = np.zeros(len(d), dtype=int)
    for i in range(2, len(d)):
        if st[i] == 0 or bos[i] >= i - 1:
            continue
        g, r = C[i] > O[i] and C[i - 1] > O[i - 1], C[i] < O[i] and C[i - 1] < O[i - 1]
        if counter:   # your rule: BOS down + 2 green -> buy ; BOS up + 2 red -> sell
            out[i] = 1 if (st[i] == -1 and g) else -1 if (st[i] == 1 and r) else 0
        else:         # with the break: BOS down + 2 red -> sell ; BOS up + 2 green -> buy
            out[i] = -1 if (st[i] == -1 and r) else 1 if (st[i] == 1 and g) else 0
    return out


VARIANTS = [  # name, signal function, SL (ATR), TP (ATR)
    ("trend_pullback 1.5/3", s_trend_pullback, 1.5, 3.0),
    ("trend_pullback 2/3", s_trend_pullback, 2.0, 3.0),
    ("donchian20 2/4", s_donchian(20), 2.0, 4.0),
    ("donchian20 2/6", s_donchian(20), 2.0, 6.0),
    ("donchian55 2/4", s_donchian(55), 2.0, 4.0),
    ("london_breakout 1.5/3", s_london_breakout, 1.5, 3.0),
    ("rsi3_meanrev 3/1.5", s_rsi_meanrev, 3.0, 1.5),
    ("bos_counter (yours) 1.5/3", lambda d: _bos(d, True), 1.5, 3.0),
    ("bos_with_trend 1.5/3", lambda d: _bos(d, False), 1.5, 3.0),
]


# ------------------------------------------------------------ simulator
def simulate(d, sig, sl_k, tp_k, spread):
    """Enter at next open. One trade at a time. SL first if both hit in one bar.
    Returns list of (entry_bar_index, R)."""
    O, H, L, C, A = (d[c].to_numpy(float) for c in ("open", "high", "low", "close", "atr"))
    n, trades, i = len(d), [], 200                         # skip indicator warm-up
    while i < n - 2:
        s = sig[i]
        if s == 0 or not np.isfinite(A[i]) or A[i] <= 0:
            i += 1; continue
        e = O[i + 1] + (spread if s == 1 else 0.0)
        risk = sl_k * A[i]
        sl, tp = e - s * risk, e + s * tp_k * A[i]
        r_out, j = None, i + 1
        while j < min(n, i + 1 + MAX_HOLD):
            if s == 1:
                sl_hit, tp_hit = L[j] <= sl, H[j] >= tp
            else:
                sl_hit, tp_hit = H[j] + spread >= sl, L[j] + spread <= tp
            if sl_hit:
                r_out = -1.0; break
            if tp_hit:
                r_out = tp_k / sl_k; break
            j += 1
        if r_out is None:
            j = min(n - 1, i + MAX_HOLD)
            x = C[j] + (spread if s == -1 else 0.0)
            r_out = (x - e) * s / risk
        trades.append((i + 1, r_out))
        i = j + 1
    return trades


def stats(rs):
    rs = np.asarray(rs, float)
    if len(rs) < 2:
        return dict(n=len(rs), avg=0.0, win=0.0, pf=0.0, t=0.0)
    gp, gl = rs[rs > 0].sum(), -rs[rs < 0].sum()
    sd = rs.std(ddof=1)
    return dict(n=len(rs), avg=rs.mean(), win=100 * (rs > 0).mean(),
                pf=gp / gl if gl else 9.99, t=rs.mean() / (sd / np.sqrt(len(rs))) if sd else 0.0)


# ----------------------------------------------------------------- main
def main():
    if not KEY:
        raise SystemExit("TWELVE_DATA_KEY missing")
    data = {}
    for s in SYMBOLS:
        df = fetch(s)
        if df is not None and len(df) > 1000:
            data[s] = prep(df)
            print(f"{s}: {len(df)} candles {df.time.iloc[0]:%Y-%m-%d} -> {df.time.iloc[-1]:%Y-%m-%d}")
    rows = []
    for name, fn, sl_k, tp_k in VARIANTS:
        is_all, oos_all, per_sym = [], [], {}
        for s, d in data.items():
            tr = simulate(d, fn(d), sl_k, tp_k, SPREAD.get(s, 0.0))
            cut = int(len(d) * IS_FRAC)
            ins = [r for i, r in tr if i < cut]
            oos = [r for i, r in tr if i >= cut]
            is_all += ins; oos_all += oos
            per_sym[s] = stats(oos)["avg"] if len(oos) >= 5 else None
        a, i_, o = stats(is_all + oos_all), stats(is_all), stats(oos_all)
        pos = sum(1 for v in per_sym.values() if v is not None and v > 0)
        tot = sum(1 for v in per_sym.values() if v is not None)
        good = o["avg"] > 0.05 and o["t"] > 1.5 and i_["avg"] > 0 and tot and pos / tot >= 0.6
        rows.append((o["avg"], name, a, i_, o, pos, tot, good))
    rows.sort(key=lambda r: -r[0])
    print("\nR = multiples of the stop distance. avgR > 0 is needed; t > 1.5 on out-of-sample = some evidence.\n")
    print(f"{'strategy':28s} {'trades':>6s} {'win%':>5s} {'avgR':>6s} {'PF':>5s} | {'IS avgR':>7s} {'OOS avgR':>8s} {'OOS t':>5s} {'syms+':>6s}  verdict")
    for _, name, a, i_, o, pos, tot, good in rows:
        print(f"{name:28s} {a['n']:6d} {a['win']:5.1f} {a['avg']:6.2f} {a['pf']:5.2f} | "
              f"{i_['avg']:7.2f} {o['avg']:8.2f} {o['t']:5.1f} {pos}/{tot:<4d}  {'CANDIDATE' if good else 'no edge'}")
    print("\n9 variants x several symbols were tested, so a lone good-looking line can be luck. "
          "Only trust a CANDIDATE that is positive in both halves and on most symbols.")


if __name__ == "__main__":
    main()
