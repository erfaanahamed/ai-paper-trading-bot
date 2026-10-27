"""Causal, rule-based Smart Money Concepts (SMC) research features.

These are price-action heuristics, not evidence of actual institutional orders.
All swing levels are based on prior candles to avoid look-ahead leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_smc_features(frame: pd.DataFrame, swing_window: int = 5, ob_lookback: int = 8) -> pd.DataFrame:
    """Return a copy with causal BOS/CHOCH, FVG, OB and liquidity-sweep features."""
    if frame is None or frame.empty or not {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        out = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        for col in ["SMC_BOS", "SMC_CHOCH", "SMC_FVG", "SMC_OB", "SMC_OB_Top", "SMC_OB_Bottom", "SMC_OB_Retest", "SMC_LiquiditySweep"]:
            out[col] = np.nan if col.endswith(("Top", "Bottom")) else 0
        return out
    out = frame.copy()
    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")
    close = pd.to_numeric(out["Close"], errors="coerce")
    opn = pd.to_numeric(out["Open"], errors="coerce")
    # Prior-only levels; current candle never defines its own breakout level.
    prior_high = high.rolling(swing_window, min_periods=swing_window).max().shift(1)
    prior_low = low.rolling(swing_window, min_periods=swing_window).min().shift(1)
    bos_up = (close > prior_high) & prior_high.notna()
    bos_down = (close < prior_low) & prior_low.notna()
    # CHOCH heuristic: a structure break opposite to the last observed break direction.
    direction = np.zeros(len(out), dtype=int)
    last_dir = 0
    for i, (up, down) in enumerate(zip(bos_up.fillna(False), bos_down.fillna(False))):
        if up:
            direction[i] = 1 if last_dir == -1 else 0
            last_dir = 1
        elif down:
            direction[i] = -1 if last_dir == 1 else 0
            last_dir = -1
    out["SMC_BOS"] = np.select([bos_up, bos_down], [1, -1], default=0).astype(int)
    out["SMC_CHOCH"] = direction
    # Three-candle imbalance, confirmed only when the third candle has closed.
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)
    out["SMC_FVG"] = np.select([fvg_up.fillna(False), fvg_down.fillna(False)], [1, -1], default=0).astype(int)
    # Liquidity sweep: pierce a prior range boundary then close back inside.
    sweep_up = (low < prior_low) & (close > prior_low) & prior_low.notna()
    sweep_down = (high > prior_high) & (close < prior_high) & prior_high.notna()
    out["SMC_LiquiditySweep"] = np.select([sweep_up.fillna(False), sweep_down.fillna(False)], [1, -1], default=0).astype(int)
    # Order-block heuristic: on a structure break, find the latest opposite candle in prior lookback.
    tops = np.full(len(out), np.nan); bottoms = np.full(len(out), np.nan); ob_dir = np.zeros(len(out), dtype=int)
    for i in range(len(out)):
        side = 1 if bool(bos_up.iloc[i]) else (-1 if bool(bos_down.iloc[i]) else 0)
        if side == 0:
            continue
        start = max(0, i - ob_lookback)
        found = None
        for j in range(i - 1, start - 1, -1):
            is_bear = opn.iloc[j] > close.iloc[j]
            is_bull = close.iloc[j] > opn.iloc[j]
            if (side == 1 and is_bear) or (side == -1 and is_bull):
                found = j; break
        if found is not None:
            tops[i] = max(opn.iloc[found], close.iloc[found], high.iloc[found])
            bottoms[i] = min(opn.iloc[found], close.iloc[found], low.iloc[found])
            ob_dir[i] = side
    out["SMC_OB"] = ob_dir
    out["SMC_OB_Top"] = tops
    out["SMC_OB_Bottom"] = bottoms
    # Carry the most recent identified zone forward until price closes beyond it.
    active_top = np.nan; active_bottom = np.nan; active_dir = 0
    retest = np.zeros(len(out), dtype=int)
    for i in range(len(out)):
        if ob_dir[i]:
            active_top, active_bottom, active_dir = tops[i], bottoms[i], ob_dir[i]
        if active_dir == 1 and np.isfinite(active_bottom) and close.iloc[i] < active_bottom:
            active_top = active_bottom = np.nan; active_dir = 0
        elif active_dir == -1 and np.isfinite(active_top) and close.iloc[i] > active_top:
            active_top = active_bottom = np.nan; active_dir = 0
        elif active_dir and i > 0 and low.iloc[i] <= active_top and high.iloc[i] >= active_bottom:
            # Do not count the candle that created the zone as a retest.
            if not ob_dir[i]: retest[i] = active_dir
    out["SMC_OB_Retest"] = retest
    out["SMC_Active_OB_Top"] = np.nan
    out["SMC_Active_OB_Bottom"] = np.nan
    active_top = active_bottom = np.nan; active_dir = 0
    for i in range(len(out)):
        if ob_dir[i]: active_top, active_bottom, active_dir = tops[i], bottoms[i], ob_dir[i]
        if active_dir == 1 and np.isfinite(active_bottom) and close.iloc[i] < active_bottom: active_top = active_bottom = np.nan; active_dir = 0
        elif active_dir == -1 and np.isfinite(active_top) and close.iloc[i] > active_top: active_top = active_bottom = np.nan; active_dir = 0
        out.iat[i, out.columns.get_loc("SMC_Active_OB_Top")] = active_top
        out.iat[i, out.columns.get_loc("SMC_Active_OB_Bottom")] = active_bottom
    return out


def smc_signal(last) -> str:
    """Return BUY/SELL only for a confluence of causal SMC clues; otherwise WAIT."""
    def val(key):
        try:
            x = float(last.get(key, 0))
            return x if np.isfinite(x) else 0.0
        except (TypeError, ValueError):
            return 0.0
    bullish = (val("SMC_OB_Retest") == 1) or (val("SMC_LiquiditySweep") == 1 and val("SMC_CHOCH") == 1)
    bearish = (val("SMC_OB_Retest") == -1) or (val("SMC_LiquiditySweep") == -1 and val("SMC_CHOCH") == -1)
    # A recent FVG/BOS may support an existing retest/sweep but cannot trigger alone.
    bullish = bullish or (val("SMC_BOS") == 1 and val("SMC_FVG") == 1 and val("SMC_OB") == 1)
    bearish = bearish or (val("SMC_BOS") == -1 and val("SMC_FVG") == -1 and val("SMC_OB") == -1)
    if bullish and not bearish: return "BUY"
    if bearish and not bullish: return "SELL"
    return "WAIT"
