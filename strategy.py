"""
Break-of-structure (BOS) + 2-candle strategy on 15m candles.  Pure logic, no I/O.

Rules (as described by Pratik)
  * BOS down = a candle CLOSES below the last confirmed swing low   -> structure "down"
  * BOS up   = a candle CLOSES above the last confirmed swing high  -> structure "up"
  * structure down + two green candles in a row (both after the BOS) -> BUY
  * structure up   + two red candles in a row   (both after the BOS) -> SELL
  * fixed 0.01 lot, SL $2, TP $4  (price distance is computed by the caller)

Swing = fractal: a high (low) higher (lower) than the SWING_N candles on each side.
A swing is only "known" SWING_N candles after it forms -> no look-ahead.
"""
import numpy as np

SWING_N = 3


def structure(df, n=SWING_N):
    """Per-candle structure state (+1 up, -1 down, 0 none) and index of the BOS candle."""
    H, L, C = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    N = len(df)
    high_conf, low_conf = {}, {}
    for i in range(n, N - n):
        if H[i] > H[i - n:i].max() and H[i] >= H[i + 1:i + n + 1].max():
            high_conf[i + n] = H[i]
        if L[i] < L[i - n:i].min() and L[i] <= L[i + 1:i + n + 1].min():
            low_conf[i + n] = L[i]
    state = np.zeros(N, dtype=int)
    bos = np.full(N, -1, dtype=int)
    sh = sl = None                     # active, not-yet-broken swing levels
    cur, cur_bos = 0, -1
    for j in range(N):
        if j in high_conf:
            sh = high_conf[j]
        if j in low_conf:
            sl = low_conf[j]
        if sl is not None and C[j] < sl:
            cur, cur_bos, sl = -1, j, None
        elif sh is not None and C[j] > sh:
            cur, cur_bos, sh = 1, j, None
        state[j], bos[j] = cur, cur_bos
    return state, bos


def signal_at(O, C, state, bos, i, fresh_only=True):
    """Signal on the CLOSE of candle i: 'BUY', 'SELL' or None."""
    if i < 2 or state[i] == 0 or bos[i] >= i - 1:      # both candles must be after the BOS
        return None
    green = lambda k: C[k] > O[k]
    red = lambda k: C[k] < O[k]
    if state[i] == -1 and green(i) and green(i - 1):
        if fresh_only and i - 2 > bos[i] and green(i - 2):
            return None                                 # 3rd+ green in a row: skip
        return "BUY"
    if state[i] == 1 and red(i) and red(i - 1):
        if fresh_only and i - 2 > bos[i] and red(i - 2):
            return None
        return "SELL"
    return None


def signal(df, n=SWING_N, fresh_only=True):
    """df = CLOSED candles only, columns open/high/low/close, oldest first."""
    if len(df) < 30:
        return None
    O, C = df["open"].to_numpy(float), df["close"].to_numpy(float)
    state, bos = structure(df, n)
    i = len(df) - 1
    side = signal_at(O, C, state, bos, i, fresh_only)
    if not side:
        return None
    return {"side": side, "candle_index": i, "bos_index": int(bos[i]),
            "structure": "down" if state[i] == -1 else "up"}


def backtest(df, sl_dist, tp_dist, spread=0.0, sl_usd=2.0, tp_usd=4.0,
             n=SWING_N, fresh_only=True):
    """Enter at next candle open. One trade at a time. If SL and TP hit in the
    same candle, SL is assumed first (conservative). Spread is paid on entry."""
    O, H, L, C = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    state, bos = structure(df, n)
    trades, pos = [], None
    for i in range(len(df)):
        if pos is not None and i >= pos["i"]:
            if pos["side"] == "BUY":
                sl_hit, tp_hit = L[i] <= pos["sl"], H[i] >= pos["tp"]
            else:
                sl_hit, tp_hit = H[i] + spread >= pos["sl"], L[i] + spread <= pos["tp"]
            if sl_hit or tp_hit:
                trades.append((pos["side"], pos["i"], i, "SL" if sl_hit else "TP"))
                pos = None
        if pos is None and i + 1 < len(df):
            s = signal_at(O, C, state, bos, i, fresh_only)
            if s:
                e = O[i + 1] + (spread if s == "BUY" else 0.0)
                d = 1 if s == "BUY" else -1
                pos = {"side": s, "i": i + 1, "sl": e - d * sl_dist, "tp": e + d * tp_dist}
    wins = sum(1 for t in trades if t[3] == "TP")
    losses = len(trades) - wins
    eq, peak, mdd = 0.0, 0.0, 0.0
    for t in trades:
        eq += tp_usd if t[3] == "TP" else -sl_usd
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    gp, gl = wins * tp_usd, losses * sl_usd
    return {"trades": len(trades), "wins": wins, "losses": losses,
            "win_rate": round(100 * wins / len(trades), 1) if trades else 0.0,
            "net_usd": round(gp - gl, 2),
            "profit_factor": round(gp / gl, 2) if gl else None,
            "max_drawdown_usd": round(mdd, 2),
            "breakeven_win_rate": round(100 * sl_usd / (sl_usd + tp_usd), 1)}
