"""
Paper-trading bot (no broker). Runs once per call (GitHub Actions every 15 min).
Gets 15m candles from Twelve Data, applies the BOS + 2-candle strategy, and records
simulated trades (0.01 lot, SL $2, TP $4) in paper/state.json.

Env: TWELVE_DATA_KEY (secret), SYMBOLS (default "EURUSD,XAUUSD")
Spreads below are ASSUMED typical values - edit to match your broker.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from strategy import signal

KEY = os.environ.get("TWELVE_DATA_KEY", "")
SYMBOLS = [s.strip() for s in os.environ.get("SYMBOLS", "EURUSD,XAUUSD").split(",") if s.strip()]
LOTS, SL_USD, TP_USD = 0.01, 2.0, 4.0
CONTRACT = {"EURUSD": 100000, "GBPUSD": 100000, "AUDUSD": 100000, "NZDUSD": 100000,
            "XAUUSD": 100, "XAGUSD": 5000}
SPREAD = {"EURUSD": 0.00012, "GBPUSD": 0.00015, "AUDUSD": 0.00015, "NZDUSD": 0.00018,
          "XAUUSD": 0.25, "XAGUSD": 0.03}
ENTRY_WINDOW_MIN = 20          # ignore signals whose candle closed longer ago than this
STATE_PATH = "paper/state.json"
MAX_TRADES = 2000
BAR = timedelta(minutes=15)


def log(*a):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"trades": [], "last_signal": {}}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    state["trades"] = state["trades"][-MAX_TRADES:]
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)


def fetch(sym):
    js = requests.get("https://api.twelvedata.com/time_series",
                      params={"symbol": sym[:3] + "/" + sym[3:], "interval": "15min",
                              "outputsize": 300, "timezone": "UTC", "apikey": KEY},
                      timeout=60).json()
    if "values" not in js:
        raise RuntimeError(js.get("message", "no data"))
    df = pd.DataFrame(js["values"]).iloc[::-1].reset_index(drop=True)
    df["time"] = pd.to_datetime(df["datetime"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c])
    return df[["time", "open", "high", "low", "close"]]


def process(state, sym, df_all, now):
    """Update open trades and look for a new signal. `now` is naive UTC."""
    out = []
    df = df_all[df_all["time"] + BAR <= now].reset_index(drop=True)     # closed candles only
    if len(df) < 30:
        return [f"{sym}: not enough candles"]
    spread = SPREAD.get(sym, 0.0)

    for t in state["trades"]:                                            # 1) resolve open trades
        if t["symbol"] != sym or t["status"] != "OPEN":
            continue
        for _, c in df[df["time"] >= pd.Timestamp(t["open_time"])].iterrows():
            if t["side"] == "BUY":
                sl_hit, tp_hit = c["low"] <= t["sl"], c["high"] >= t["tp"]
            else:
                sl_hit, tp_hit = c["high"] + spread >= t["sl"], c["low"] + spread <= t["tp"]
            if sl_hit or tp_hit:
                t["status"] = "SL" if sl_hit else "TP"                  # both in one candle -> SL
                t["pnl"] = -SL_USD if sl_hit else TP_USD
                t["close_time"] = str(c["time"] + BAR)
                if sl_hit and tp_hit:
                    t["note"] = "SL and TP in same candle, SL assumed"
                out.append(f"{sym}: trade {t['id']} closed {t['status']} ({t['pnl']:+.0f}$)")
                break

    sig = signal(df)                                                     # 2) new signal?
    if not sig:
        return out + [f"{sym}: no signal"]
    sig_time = df["time"].iloc[-1]
    close_time = sig_time + BAR
    age = (now - close_time).total_seconds() / 60
    if age > ENTRY_WINDOW_MIN:
        return out + [f"{sym}: signal is old ({age:.0f} min), skipped"]
    if state["last_signal"].get(sym) == str(sig_time):
        return out + [f"{sym}: signal already taken"]
    if any(t["symbol"] == sym and t["status"] == "OPEN" for t in state["trades"]):
        return out + [f"{sym}: {sig['side']} signal but a trade is already open"]

    dist_sl = SL_USD / (LOTS * CONTRACT[sym])
    dist_tp = TP_USD / (LOTS * CONTRACT[sym])
    px = float(df["close"].iloc[-1])
    if sig["side"] == "BUY":
        e = px + spread; sl, tp = e - dist_sl, e + dist_tp
    else:
        e = px; sl, tp = e + dist_sl, e - dist_tp
    state["trades"].append({
        "id": f"{sym}-{sig_time:%Y%m%d%H%M}", "symbol": sym, "side": sig["side"],
        "structure": sig["structure"], "signal_time": str(sig_time), "open_time": str(close_time),
        "entry": round(e, 5), "sl": round(sl, 5), "tp": round(tp, 5), "status": "OPEN"})
    state["last_signal"][sym] = str(sig_time)
    out.append(f"{sym}: NEW {sig['side']} @ {e:.5f} SL {sl:.5f} TP {tp:.5f}")
    return out


def main():
    if not KEY:
        raise SystemExit("TWELVE_DATA_KEY missing")
    state = load_state()
    before = json.dumps(state, sort_keys=True)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for i, sym in enumerate(SYMBOLS):
        if sym not in CONTRACT:
            log(sym, "unsupported (USD-quoted symbols only):", ", ".join(CONTRACT)); continue
        if i and i % 8 == 0:
            time.sleep(61)                       # free plan: 8 credits / minute
        try:
            for line in process(state, sym, fetch(sym), now):
                log(line)
        except Exception as e:
            log(sym, "ERROR:", e)
    if json.dumps(state, sort_keys=True) != before:
        save_state(state)
        log("state saved")


if __name__ == "__main__":
    main()
