"""
NVDA Daily 9/21 EMA Bearish Crossover Analysis
===============================================
AI era: January 2023 – present
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

print("=" * 72)
print("NVDA DAILY 9/21 EMA BEARISH CROSSOVER ANALYSIS  |  AI Era 2023-present")
print("=" * 72)
print()

# ── 1. FETCH DATA ──────────────────────────────────────────────────────────────

START = "2022-10-01"   # extra buffer so EMAs are warm by Jan 2023
END   = datetime.today().strftime("%Y-%m-%d")

nvda = yf.download("NVDA", start=START, end=END, auto_adjust=True, progress=False)
if isinstance(nvda.columns, pd.MultiIndex):
    nvda.columns = nvda.columns.get_level_values(0)
nvda.index = pd.to_datetime(nvda.index)
nvda.sort_index(inplace=True)

close = nvda["Close"]

# ── 2. EMAs ────────────────────────────────────────────────────────────────────

ema9  = close.ewm(span=9,  adjust=False).mean()
ema21 = close.ewm(span=21, adjust=False).mean()

# ── 3. RSI (for context) ──────────────────────────────────────────────────────

def calc_rsi(series, window=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

rsi14 = calc_rsi(close)

# Clip to AI era
AI_START = pd.Timestamp("2023-01-01")
mask_ai  = nvda.index >= AI_START

close_ai = close[mask_ai]
ema9_ai  = ema9[mask_ai]
ema21_ai = ema21[mask_ai]
rsi_ai   = rsi14[mask_ai]

# ── 4. DETECT BEARISH CROSSOVERS (9 EMA crosses BELOW 21 EMA) ─────────────────

above_prev = (ema9_ai.shift(1) >= ema21_ai.shift(1))
below_now  = (ema9_ai < ema21_ai)
bearish_cross = above_prev & below_now

cross_dates = bearish_cross[bearish_cross].index
print(f"Bearish crossovers found (AI era): {len(cross_dates)}")
print()

# ── 5. FOR EACH CROSSOVER: DRAWDOWN + RECOVERY + 90d FORWARD RETURN ──────────

results = []

for cx_date in cross_dates:
    cx_price = float(close_ai.loc[cx_date])

    # Future prices after crossover
    future = close_ai[close_ai.index > cx_date]
    if len(future) == 0:
        continue

    # Find next BULLISH crossover (9 crosses back above 21) — marks recovery period
    future_ema9  = ema9_ai[ema9_ai.index > cx_date]
    future_ema21 = ema21_ai[ema21_ai.index > cx_date]

    bull_cross_mask = (
        (future_ema9.shift(1) <= future_ema21.shift(1)) &
        (future_ema9 > future_ema21)
    )
    bull_dates = bull_cross_mask[bull_cross_mask].index

    # Drawdown window: from crossover to next bull cross (or 252 days max)
    if len(bull_dates) > 0:
        next_bull = bull_dates[0]
        drawdown_window = close_ai.loc[cx_date:next_bull]
        recovery_days   = (next_bull - cx_date).days
        recovered_to    = float(close_ai.loc[next_bull])
    else:
        # No recovery yet — use all remaining data
        drawdown_window = future
        recovery_days   = None
        recovered_to    = None
        next_bull       = None

    # Max drawdown within window
    trough_price = float(drawdown_window.min())
    trough_date  = drawdown_window.idxmin()
    max_dd_pct   = (trough_price - cx_price) / cx_price * 100
    days_to_trough = (trough_date - cx_date).days

    # 90-day forward return from crossover
    future_90 = close_ai[close_ai.index >= cx_date]
    if len(future_90) >= 90:
        fwd90_price  = float(future_90.iloc[90])
        fwd90_return = (fwd90_price / cx_price - 1) * 100
    else:
        fwd90_return = None

    results.append({
        "Crossover Date":   cx_date.date(),
        "Price at Cross":   round(cx_price, 2),
        "Trough Date":      trough_date.date(),
        "Trough Price":     round(trough_price, 2),
        "Max DD %":         round(max_dd_pct, 1),
        "Days to Trough":   days_to_trough,
        "Recovery Date":    next_bull.date() if next_bull is not None else "Ongoing",
        "Recovery Days":    recovery_days,
        "Recovered Price":  round(recovered_to, 2) if recovered_to else None,
        "Fwd 90d %":        round(fwd90_return, 1) if fwd90_return is not None else None,
        "RSI at Cross":     round(float(rsi_ai.loc[cx_date]), 1),
    })

rdf = pd.DataFrame(results)

# ── 6. PRINT TABLE ─────────────────────────────────────────────────────────────

print(f"{'#':<3} {'Cross Date':<13} {'Cross $':>9} {'Trough $':>9} {'Max DD%':>8} {'Days↓':>7} {'Recovery':>12} {'Fwd90d%':>9} {'RSI':>6}")
print("-" * 85)
for i, r in rdf.iterrows():
    fwd_str  = f"{r['Fwd 90d %']:>8.1f}" if r["Fwd 90d %"] is not None else "     N/A"
    rec_str  = str(r["Recovery Date"])[:10] if r["Recovery Date"] != "Ongoing" else "  Ongoing"
    print(f"{i+1:<3} {str(r['Crossover Date']):<13} ${r['Price at Cross']:>8.2f} "
          f"${r['Trough Price']:>8.2f} {r['Max DD %']:>7.1f}% {r['Days to Trough']:>6}d  "
          f"{rec_str:<12} {fwd_str}% {r['RSI at Cross']:>6.1f}")

# ── 7. SUMMARY STATS ──────────────────────────────────────────────────────────

completed = rdf[rdf["Recovery Date"] != "Ongoing"].copy()
all_dd    = rdf["Max DD %"]

print()
print("=" * 72)
print("SUMMARY STATISTICS  (completed crossover episodes)")
print("=" * 72)
print()
print(f"  Total bearish crossovers (AI era):    {len(rdf)}")
print(f"  Completed (followed by bull cross):   {len(completed)}")
print(f"  Currently open/ongoing:               {len(rdf) - len(completed)}")
print()
print(f"  Average max drawdown after crossover: {all_dd.mean():.1f}%")
print(f"  Median  max drawdown after crossover: {all_dd.median():.1f}%")
print(f"  Worst drawdown:                       {all_dd.min():.1f}%")
print(f"  Mildest drawdown:                     {all_dd.max():.1f}%  (some crosses are brief)")
print()
if len(completed) > 0:
    avg_rec_days = completed["Recovery Days"].mean()
    avg_fwd90    = rdf["Fwd 90d %"].dropna().mean()
    med_fwd90    = rdf["Fwd 90d %"].dropna().median()
    print(f"  Avg days to trough:                   {rdf['Days to Trough'].mean():.0f}d")
    print(f"  Avg calendar days to recovery:        {avg_rec_days:.0f}d")
    print(f"  Avg 90d forward return from cross:    {avg_fwd90:.1f}%")
    print(f"  Median 90d forward return from cross: {med_fwd90:.1f}%")
    pct_pos = (rdf["Fwd 90d %"].dropna() > 0).mean() * 100
    print(f"  % of crosses with positive 90d fwd:  {pct_pos:.0f}%")

# ── 8. CURRENT STATUS ─────────────────────────────────────────────────────────

print()
print("=" * 72)
print("CURRENT STATUS  (as of latest available data)")
print("=" * 72)
print()

latest_date  = close_ai.index[-1]
latest_price = float(close_ai.iloc[-1])
latest_ema9  = float(ema9_ai.iloc[-1])
latest_ema21 = float(ema21_ai.iloc[-1])
latest_rsi   = float(rsi_ai.iloc[-1])
latest_gap   = latest_ema9 - latest_ema21  # negative = 9 below 21 (bearish regime)
prev_gap     = float(ema9_ai.iloc[-2]) - float(ema21_ai.iloc[-2])

print(f"  Date:          {latest_date.date()}")
print(f"  Price:         ${latest_price:.2f}")
print(f"  EMA-9:         ${latest_ema9:.2f}")
print(f"  EMA-21:        ${latest_ema21:.2f}")
print(f"  EMA gap:       {latest_gap:+.2f}  ({'9 BELOW 21 — bearish' if latest_gap < 0 else '9 ABOVE 21 — bullish'})")
print(f"  RSI (14):      {latest_rsi:.1f}")
print()

if latest_gap < 0:
    # Find when the current bearish regime started
    bearish_run = (ema9_ai < ema21_ai)
    # Walk back to find start of current run
    run_start = latest_date
    for d in reversed(list(bearish_run.index)):
        if bearish_run.loc[d]:
            run_start = d
        else:
            break
    days_in_bear = (latest_date - run_start).days
    cross_ref    = float(close_ai.loc[run_start])
    dd_so_far    = (latest_price - cross_ref) / cross_ref * 100

    # Most recent crossover
    if len(cross_dates) > 0:
        last_cx      = cross_dates[-1]
        last_cx_price = float(close_ai.loc[last_cx])
        dd_from_last = (latest_price - last_cx_price) / last_cx_price * 100
        days_since   = (latest_date - last_cx).days
        print(f"  *** CURRENTLY IN BEARISH EMA REGIME ***")
        print(f"  Last bearish crossover:  {last_cx.date()}  @ ${last_cx_price:.2f}")
        print(f"  Days since crossover:    {days_since}d")
        print(f"  Drawdown from cross:     {dd_from_last:.1f}%")
        print()

        # Project likely bottom using historical averages
        avg_dd        = all_dd.mean()
        med_dd        = all_dd.median()
        avg_days_bot  = rdf["Days to Trough"].mean()
        med_days_bot  = rdf["Days to Trough"].median()

        price_at_avg_dd  = last_cx_price * (1 + avg_dd / 100)
        price_at_med_dd  = last_cx_price * (1 + med_dd / 100)
        pct_of_avg_done  = dd_from_last / avg_dd * 100 if avg_dd != 0 else 0
        pct_of_med_done  = dd_from_last / med_dd * 100 if med_dd != 0 else 0

        print(f"  PROJECTED BOTTOM (from crossover price ${last_cx_price:.2f}):")
        print(f"    Using average DD ({avg_dd:.1f}%):  target ~${price_at_avg_dd:.0f}  ({pct_of_avg_done:.0f}% of avg drawdown already done)")
        print(f"    Using median DD  ({med_dd:.1f}%):  target ~${price_at_med_dd:.0f}  ({pct_of_med_done:.0f}% of median drawdown already done)")
        print(f"    Avg days to trough: {avg_days_bot:.0f}d  |  Median: {med_days_bot:.0f}d")
        remaining_days_avg = max(0, avg_days_bot - days_since)
        print(f"    ~{remaining_days_avg:.0f} calendar days remaining to avg trough (from today)")
else:
    print(f"  EMA-9 is ABOVE EMA-21 — no bearish regime currently active.")
    gap_to_cross = latest_ema21 - latest_ema9  # how far 9 needs to fall to trigger
    print(f"  Gap to bearish crossover: {latest_ema9 - latest_ema21:+.2f} (9 needs to drop {gap_to_cross:.2f} more to cross)")

print()
print("=" * 72)
print("OPTIMAL ENTRY TIMING: WHERE IN THE DRAWDOWN DID BEST LEAPS ENTRIES APPEAR?")
print("=" * 72)
print()

# For each completed episode, find the point where RSI first hit <35 and <30
# and note what % through the drawdown that was
print(f"  Historical drawdown pattern across all crossovers:")
print(f"  {'Episode':<22} {'Cross $':>8} {'Trough $':>9} {'DD%':>7} {'Days':>6} {'Fwd90%':>9}")
print("  " + "-" * 67)
for _, r in rdf.iterrows():
    fwd_str = f"{r['Fwd 90d %']:>8.1f}" if r["Fwd 90d %"] is not None else "     N/A"
    print(f"  {str(r['Crossover Date']):<22} ${r['Price at Cross']:>7.2f} ${r['Trough Price']:>8.2f} "
          f"{r['Max DD %']:>6.1f}% {r['Days to Trough']:>5}d {fwd_str}%")

print()

# The key question: when within a drawdown does the best LEAPS entry appear?
# Proxy: Tier 1 requires RSI<35 + >20% below 52w high + >10% below 200SMA
# Let's find the % of max-drawdown completed at the first day RSI < 35 for each episode

print(f"  Timing of RSI capitulation within each drawdown:")
print(f"  {'Episode':<13} {'Total DD%':>10} {'RSI<40 first':>13} {'% DD done at RSI<40':>20} {'RSI<35 first':>13} {'% DD done at RSI<35':>20}")
print("  " + "-" * 95)

for cx_date in cross_dates:
    cx_price = float(close_ai.loc[cx_date])
    future_close = close_ai[close_ai.index >= cx_date]
    future_rsi   = rsi_ai[rsi_ai.index >= cx_date]

    if len(future_close) < 5:
        continue

    # Total drawdown for this episode (use window to next bull cross or 200d)
    future_ema9_ep  = ema9_ai[ema9_ai.index > cx_date]
    future_ema21_ep = ema21_ai[ema21_ai.index > cx_date]
    bull_mask = (future_ema9_ep.shift(1) <= future_ema21_ep.shift(1)) & (future_ema9_ep > future_ema21_ep)
    bull_ep = bull_mask[bull_mask].index
    if len(bull_ep) > 0:
        window_close = close_ai.loc[cx_date:bull_ep[0]]
    else:
        window_close = future_close.iloc[:200]

    trough_p  = float(window_close.min())
    trough_d  = window_close.idxmin()
    total_dd  = (trough_p - cx_price) / cx_price * 100

    # First day RSI drops below 40 and 35
    rsi40_dates = future_rsi[(future_rsi < 40) & (future_rsi.index >= cx_date) & (future_rsi.index <= trough_d)]
    rsi35_dates = future_rsi[(future_rsi < 35) & (future_rsi.index >= cx_date) & (future_rsi.index <= trough_d)]

    rsi40_str = "never"
    rsi35_str = "never"
    pct40_str = "—"
    pct35_str = "—"

    if len(rsi40_dates) > 0:
        first40 = rsi40_dates.index[0]
        price40 = float(close_ai.loc[first40])
        dd40    = (price40 - cx_price) / cx_price * 100
        pct_done40 = dd40 / total_dd * 100 if total_dd != 0 else 0
        rsi40_str = str(first40.date())
        pct40_str = f"{pct_done40:.0f}% of DD done"

    if len(rsi35_dates) > 0:
        first35 = rsi35_dates.index[0]
        price35 = float(close_ai.loc[first35])
        dd35    = (price35 - cx_price) / cx_price * 100
        pct_done35 = dd35 / total_dd * 100 if total_dd != 0 else 0
        rsi35_str = str(first35.date())
        pct35_str = f"{pct_done35:.0f}% of DD done"

    print(f"  {str(cx_date.date()):<13} {total_dd:>9.1f}%  {rsi40_str:<14} {pct40_str:<20} {rsi35_str:<14} {pct35_str:<20}")
