"""
Backtest of Pratik's 9-EMA retest trend strategy (same rules as EmaRetestEA.mq5).

  BUY : EMA9 > EMA21 > EMA50 and close > EMA50; a strong bullish candle dips to/through the
        9 EMA and closes above it -> enter at the next candle open.  SELL = mirror image.
  SL  : below (above) that candle's low (high) -/+ buffer.
  TP  : nearest swing high (low) ahead, taken ONLY if reward >= MIN_RR x risk.
  Limits: 10 trades/day, 1 open trade, max risk per trade, daily loss limit.

Env: TWELVE_DATA_KEY (secret), SYMBOLS (default XAUUSD), TF (15min | 5min | 1h), PAGES (default 3)
Spreads / buffers below are ASSUMPTIONS - edit them to match your account.
"""
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

KEY = os.environ.get("TWELVE_DATA_KEY", "")
SYMBOLS = [s.strip() for s in os.environ.get("SYMBOLS", "XAUUSD").split(",") if s.strip()]
TF = os.environ.get("TF", "15min")
PAGES = int(os.environ.get("PAGES", "3"))          # 5000 candles per page

LOTS = 0.01
CONTRACT = {"XAUUSD": 100, "XAGUSD": 5000, "EURUSD": 100000, "GBPUSD": 100000,
            "AUDUSD": 100000, "NZDUSD": 100000}
SPREAD = {"XAUUSD": 0.30, "XAGUSD": 0.03, "EURUSD": 0.00012, "GBPUSD": 0.00015,
          "AUDUSD": 0.00015, "NZDUSD": 0.00018}
BUFFER = {"XAUUSD": 1.5, "XAGUSD": 0.05, "EURUSD": 0.0002, "GBPUSD": 0.0002,
          "AUDUSD": 0.0002, "NZDUSD": 0.0002}

N_SWING, LOOKBACK = 3, 150
MIN_BODY_PCT, MIN_RANGE_ATR = 60.0, 0.8
MAX_TRADES_DAY, DAILY_LOSS_LIMIT, MAX_RISK_MONEY = 10, 20.0, 10.0
IS_FRAC = 0.6


def fetch(sym):
    frames, end = [], None
    for _ in range(PAGES):
        p = {"symbol": sym[:3] + "/" + sym[3:], "interval": TF, "outputsize": 5000,
             "timezone": "UTC", "apikey": KEY}
        if end:
            p["end_date"] = end
        js = requests.get("https://api.twelvedata.com/time_series", params=p, timeout=90).json()
        time.sleep(8)
        if "values" not in js:
            print(f"  {sym}: {js.get('message', 'no data')}")
            break
        d = pd.DataFrame(js["values"])
        frames.append(d)
        end = d["datetime"].iloc[-1]
        if len(d) < 4000:
            break
    if not frames:
        return None
    df = pd.concat(frames).drop_duplicates("datetime")
    df["time"] = pd.to_datetime(df["datetime"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c])
    return df.sort_values("time").reset_index(drop=True)[["time", "open", "high", "low", "close"]]


def prep(df):
    d = df.copy()
    pc = d["close"].shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()                       # MT5 iATR = simple average of TR
    for n in (9, 21, 50):
        d[f"e{n}"] = d.close.ewm(span=n, adjust=False).mean()
    d["day"] = d.time.dt.date
    return d


def swing_flags(H, L, n=N_SWING):
    N = len(H)
    sh, sl = np.zeros(N, bool), np.zeros(N, bool)
    for m in range(n, N - n):
        if H[m] > H[m - n:m].max() and H[m] > H[m + 1:m + n + 1].max():
            sh[m] = True
        if L[m] < L[m - n:m].min() and L[m] < L[m + 1:m + n + 1].min():
            sl[m] = True
    return sh, sl


def run(d, sym, cfg):
    O, H, L, C = (d[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    A, E9, E21, E50 = (d[c].to_numpy(float) for c in ("atr", "e9", "e21", "e50"))
    days = d["day"].to_numpy()
    N, mult = len(d), LOTS * CONTRACT[sym]
    spread, buf = SPREAD[sym], cfg.get("buffer", BUFFER[sym])
    sh, sl_f = swing_flags(H, L)
    trades, skip = [], defaultdict(int)
    n_day, pnl_day = defaultdict(int), defaultdict(float)
    i, busy_until = 60, -1
    while i < N - 2:
        if i <= busy_until or not np.isfinite(A[i]):
            i += 1; continue
        rng = H[i] - L[i]
        if rng <= 0:
            i += 1; continue
        body = abs(C[i] - O[i])
        if not (body / rng * 100 >= MIN_BODY_PCT and rng >= MIN_RANGE_ATR * A[i]):
            i += 1; continue
        up = E9[i] > E21[i] > E50[i] and C[i] > E50[i]
        dn = E9[i] < E21[i] < E50[i] and C[i] < E50[i]
        if up and C[i] > O[i] and L[i] <= E9[i] and C[i] > E9[i]:
            d_ = 1
        elif dn and C[i] < O[i] and H[i] >= E9[i] and C[i] < E9[i]:
            d_ = -1
        else:
            i += 1; continue
        day = days[i + 1]
        if n_day[day] >= MAX_TRADES_DAY or pnl_day[day] <= -DAILY_LOSS_LIMIT:
            skip["daily limit"] += 1; i += 1; continue
        entry = O[i + 1] + (spread if d_ == 1 else 0.0)
        sl = (L[i] - buf) if d_ == 1 else (H[i] + buf)
        risk = abs(entry - sl)
        if risk <= 0:
            i += 1; continue
        if risk * mult > MAX_RISK_MONEY:
            skip["risk too big"] += 1; i += 1; continue
        if cfg["mode"] == "swing":
            lo, hi = max(N_SWING, i + 1 - LOOKBACK), i - N_SWING
            tp = None
            if hi >= lo:
                if d_ == 1:
                    m = np.arange(lo, hi + 1)[sh[lo:hi + 1] & (H[lo:hi + 1] > entry)]
                    tp = H[m].min() if m.size else None
                else:
                    m = np.arange(lo, hi + 1)[sl_f[lo:hi + 1] & (L[lo:hi + 1] < entry)]
                    tp = L[m].max() if m.size else None
            if tp is None:
                skip["no swing ahead"] += 1; i += 1; continue
            if abs(tp - entry) / risk < cfg["min_rr"]:
                skip["RR too low"] += 1; i += 1; continue
        else:
            tp = entry + d_ * cfg["rr"] * risk
        res, j = None, i + 1
        while j < N:
            if d_ == 1:
                s_hit, t_hit = L[j] <= sl, H[j] >= tp
            else:
                s_hit, t_hit = H[j] + spread >= sl, L[j] + spread <= tp
            if s_hit or t_hit:
                level = sl if s_hit else tp
                res = (level - entry) * d_ * mult
                break
            j += 1
        if res is None:
            skip["unfinished at end"] += 1; break
        n_day[day] += 1
        pnl_day[days[j]] += res
        trades.append(dict(i=i + 1, pnl=res, r=res / (risk * mult), month=str(d.time.iloc[i + 1])[:7]))
        busy_until = j
        i = j + 1
    return trades, skip


def summarize(tr, N):
    if not tr:
        return dict(n=0, win=0, avg_r=0, pf=0, net=0, dd=0, is_net=0, oos_net=0)
    p = np.array([t["pnl"] for t in tr])
    gp, gl = p[p > 0].sum(), -p[p < 0].sum()
    eq = np.cumsum(p)
    cut = int(N * IS_FRAC)
    return dict(n=len(p), win=100 * (p > 0).mean(), avg_r=np.mean([t["r"] for t in tr]),
                pf=gp / gl if gl else 9.99, net=p.sum(), dd=(np.maximum.accumulate(eq) - eq).max(),
                is_net=sum(t["pnl"] for t in tr if t["i"] < cut),
                oos_net=sum(t["pnl"] for t in tr if t["i"] >= cut))


VARIANTS = [
    ("BASE: swing target, RR>=3", dict(mode="swing", min_rr=3.0)),
    ("swing target, RR>=2", dict(mode="swing", min_rr=2.0)),
    ("fixed 3R target (no swing)", dict(mode="fixed", rr=3.0)),
    ("fixed 2R target (no swing)", dict(mode="fixed", rr=2.0)),
    ("BASE, buffer x0.67", dict(mode="swing", min_rr=3.0, buffer_mult=0.67)),
    ("BASE, buffer x1.33", dict(mode="swing", min_rr=3.0, buffer_mult=1.33)),
]


def main():
    if not KEY:
        raise SystemExit("TWELVE_DATA_KEY missing")
    for sym in SYMBOLS:
        if sym not in CONTRACT:
            print(sym, "unsupported; use:", ", ".join(CONTRACT)); continue
        df = fetch(sym)
        if df is None or len(df) < 500:
            print(sym, "not enough data"); continue
        d = prep(df)
        days = (d.time.iloc[-1] - d.time.iloc[0]).days
        print(f"\n=== {sym} {TF}: {len(d)} candles, {d.time.iloc[0]:%Y-%m-%d} -> {d.time.iloc[-1]:%Y-%m-%d} ({days} days) ===")
        print(f"spread {SPREAD[sym]} | buffer {BUFFER[sym]} | lots {LOTS} | $ per trade at 0.01 lot")
        print(f"{'variant':30s} {'trades':>6s} {'win%':>5s} {'avgR':>6s} {'PF':>5s} {'net$':>8s} {'maxDD$':>7s} {'IS$':>7s} {'OOS$':>7s}")
        base_skip, base_tr = None, None
        for name, cfg in VARIANTS:
            c = dict(cfg)
            if "buffer_mult" in c:
                c["buffer"] = BUFFER[sym] * c.pop("buffer_mult")
            tr, skip = run(d, sym, c)
            s = summarize(tr, len(d))
            print(f"{name:30s} {s['n']:6d} {s['win']:5.1f} {s['avg_r']:6.2f} {s['pf']:5.2f} "
                  f"{s['net']:8.1f} {s['dd']:7.1f} {s['is_net']:7.1f} {s['oos_net']:7.1f}")
            if base_skip is None:
                base_skip, base_tr = skip, tr
        print("\nBASE skipped signals:", dict(base_skip))
        by_m = defaultdict(float)
        for t in base_tr:
            by_m[t["month"]] += t["pnl"]
        print("BASE net $ by month:", {k: round(v, 1) for k, v in sorted(by_m.items())})
    print("\nReading it: avgR > 0 and PF > 1 on BOTH IS and OOS = some evidence. Trade count matters:"
          " under ~100 trades is too few to trust. Fills at SL/TP ignore gaps and slippage.")


if __name__ == "__main__":
    main()
