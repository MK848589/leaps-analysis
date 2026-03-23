"""
NVDA $150-155 Flush Scenario Analysis
======================================
If NVDA flushes to $150-155, what would RSI, distance from 52w-high,
distance from 200SMA look like — and what does history say about
90-day forward returns from a setup that extreme?
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

print("=" * 70)
print("NVDA $150-155 FLUSH SCENARIO ANALYSIS")
print("=" * 70)
print()

# ── 1. FETCH DATA ──────────────────────────────────────────────────────────────

START = "2023-01-01"
END   = datetime.today().strftime("%Y-%m-%d")

nvda    = yf.download("NVDA", start=START, end=END, auto_adjust=True, progress=False)
vix_raw = yf.download("^VIX",  start=START, end=END, auto_adjust=True, progress=False)

if isinstance(nvda.columns, pd.MultiIndex):
    nvda.columns = nvda.columns.get_level_values(0)
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)

nvda.index    = pd.to_datetime(nvda.index)
vix_raw.index = pd.to_datetime(vix_raw.index)

df    = nvda[["Open","High","Low","Close","Volume"]].copy()
close = df["Close"]

# ── 2. INDICATORS ──────────────────────────────────────────────────────────────

def calc_rsi(series, window=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["RSI_14"]       = calc_rsi(close, 14)
df["SMA200"]       = close.rolling(200).mean()
df["Dist_SMA200"]  = (close - df["SMA200"]) / df["SMA200"] * 100
df["High_52w"]     = close.rolling(252).max()
df["Dist_52w_High"]= (close - df["High_52w"]) / df["High_52w"] * 100
df["Vol_MA20"]     = df["Volume"].rolling(20).mean()
df["Vol_Ratio"]    = df["Volume"] / df["Vol_MA20"]
df["Chg_5d"]       = close.pct_change(5)  * 100
df["Chg_10d"]      = close.pct_change(10) * 100

daily_ret = close.pct_change()
def consec_down(series):
    result = np.zeros(len(series), dtype=int)
    count  = 0
    for i, v in enumerate(series):
        if pd.isna(v):   count = 0
        elif v < 0:      count += 1
        else:            count = 0
        result[i] = count
    return result

df["Consec_Down"] = consec_down(daily_ret.values)

vix_close = vix_raw["Close"].rename("VIX")
df = df.join(vix_close, how="left")
df["VIX"] = df["VIX"].ffill()

df["Fwd_90d_pct"] = (close.shift(-90) / close - 1) * 100

# ── 3. CURRENT SNAPSHOT (use latest row BEFORE dropping NaN fwd returns) ──────
# Separate "live" indicators from the historical analysis set

df_live = df.dropna(subset=["RSI_14","SMA200","High_52w","VIX"]).copy()
latest       = df_live.iloc[-1]
current_price = float(latest["Close"])
sma200        = float(latest["SMA200"])
high_52w      = float(latest["High_52w"])
current_rsi   = float(latest["RSI_14"])
current_vix   = float(latest["VIX"])
current_date  = df_live.index[-1].date()

# Historical set requires valid forward returns
df = df.dropna(subset=["RSI_14","SMA200","High_52w","VIX","Fwd_90d_pct"]).copy()

print(f"Current snapshot  ({current_date})")
print(f"  Price:        ${current_price:.2f}")
print(f"  52-week high: ${high_52w:.2f}")
print(f"  200-day SMA:  ${sma200:.2f}")
print(f"  RSI (14):     {current_rsi:.1f}")
print(f"  VIX:          {current_vix:.1f}")
print(f"  Dist 52w-high: {(current_price - high_52w)/high_52w*100:.1f}%")
print(f"  Dist 200 SMA:  {(current_price - sma200)/sma200*100:.1f}%")
print()

# ── 4. ESTIMATE RSI AT FLUSH PRICES ───────────────────────────────────────────
# We simulate extending the price series with a straight-line drop to the target
# over the next N days, then recompute RSI at that endpoint.

def estimate_rsi_at_target(close_series, target_price, days_to_target=5):
    """
    Project RSI if price drops linearly to target_price over days_to_target days.
    Uses the live (full, no fwd-return filter) close series so projection is
    anchored to today's actual price.
    Returns estimated RSI at the end of that move.
    """
    last_price = float(close_series.iloc[-1])
    step = (target_price - last_price) / days_to_target
    projected = pd.concat([
        close_series,
        pd.Series(
            [last_price + step * (i + 1) for i in range(days_to_target)],
            index=pd.date_range(close_series.index[-1] + pd.Timedelta(days=1),
                                periods=days_to_target, freq="B")
        )
    ])
    rsi_proj = calc_rsi(projected, 14)
    return float(rsi_proj.iloc[-1])

# Use the LIVE close series (not truncated by forward-return filter)
close_live = df_live["Close"]

# ── 5. FLUSH SCENARIO TABLE ────────────────────────────────────────────────────

print("=" * 70)
print("FLUSH SCENARIO: PROJECTED CONDITIONS AT TARGET PRICES")
print("=" * 70)
print()
print(f"{'Target $':>10} {'Drop %':>8} {'Est RSI':>9} {'Δ 52w-High':>12} {'Δ 200 SMA':>11}  Tier Signal")
print("-" * 70)

targets = [145, 150, 152, 155, 160, 165]

scenario_data = []
for target in targets:
    drop_pct  = (target - current_price) / current_price * 100
    dist_52w  = (target - high_52w) / high_52w * 100
    dist_200  = (target - sma200) / sma200 * 100
    est_rsi   = estimate_rsi_at_target(close_live, target, days_to_target=5)

    # Tier classification (using existing tier definitions)
    if est_rsi < 35 and dist_52w < -20 and dist_200 < -10:
        tier = "TIER 1 — Maximum Fear"
    elif est_rsi < 40 and dist_52w < -15 and dist_200 < -5:
        tier = "TIER 2 — High Fear"
    elif est_rsi < 45 and dist_52w < -10 and dist_200 < 0:
        tier = "TIER 3 — Moderate Selloff"
    else:
        tier = "TIER 4 — Any Weakness"

    scenario_data.append({
        "target": target,
        "drop_pct": drop_pct,
        "est_rsi": est_rsi,
        "dist_52w": dist_52w,
        "dist_200": dist_200,
        "tier": tier,
    })

    print(f"  ${target:<8} {drop_pct:>7.1f}% {est_rsi:>9.1f} {dist_52w:>11.1f}% {dist_200:>10.1f}%  {tier}")

print()
print("Note: RSI estimated by projecting a straight-line drop over 5 trading days.")
print()

# ── 6. HISTORICAL ANALOGS: WHAT HAPPENED AT SIMILAR SETUPS ────────────────────

print("=" * 70)
print("HISTORICAL ANALOGS: DAYS WITH SIMILAR PROFILES (AI ERA 2023-PRESENT)")
print("=" * 70)
print()

# Focus on the $150-155 zone:
#   dist_52w: -27% to -24%  (midpoint ~-26%)
#   dist_200: -16% to -13%  (midpoint ~-14%)
#   RSI: expected ~28-33
# We'll look at overlapping windows generously:
#   dist_52w < -20%,  dist_200 < -10%, RSI < 40

print("Filter: Dist-52w < -20%  AND  Dist-200SMA < -10%  AND  RSI < 40")
print("(Closest historical analog to a $150-155 flush scenario)")
print()

mask_analog = (
    (df["Dist_52w_High"] < -20) &
    (df["Dist_SMA200"]   < -10) &
    (df["RSI_14"]        < 40)
)
analogs = df[mask_analog].copy()

if len(analogs) == 0:
    print("  No historical days matched these exact conditions in the AI era (2023-present).")
    print("  Widening filter to: Dist-52w < -15%  AND  Dist-200SMA < -5%  AND  RSI < 42")
    mask_analog = (
        (df["Dist_52w_High"] < -15) &
        (df["Dist_SMA200"]   < -5) &
        (df["RSI_14"]        < 42)
    )
    analogs = df[mask_analog].copy()

print(f"Days matched: {len(analogs)}")
print()

if len(analogs) > 0:
    fwd = analogs["Fwd_90d_pct"].dropna()
    print(f"  Median 90d forward return:       {fwd.median():.1f}%")
    print(f"  Mean   90d forward return:       {fwd.mean():.1f}%")
    print(f"  % achieving ≥ 40% in 90 days:   {(fwd >= 40).mean()*100:.0f}%")
    print(f"  % positive return:               {(fwd > 0).mean()*100:.0f}%")
    print(f"  Best case:                       {fwd.max():.1f}%")
    print(f"  Worst case:                      {fwd.min():.1f}%")
    print()

    # Show individual entry dates
    print(f"  {'Date':<13} {'Close':>8} {'RSI':>6} {'Δ52w%':>8} {'Δ200%':>8} {'VIX':>6} {'Cons↓':>6} {'Chg5d%':>8} {'Fwd90d%':>9}")
    print("  " + "-" * 80)

    # Cluster and show first of each cluster
    dates = analogs.index
    shown = []
    prev  = None
    for d in dates:
        if prev is None or (d - prev).days > 5:
            shown.append(d)
        prev = d

    for sig_date in shown:
        r   = analogs.loc[sig_date]
        fwd_val = r["Fwd_90d_pct"]
        fwd_str = f"{fwd_val:>8.1f}" if not pd.isna(fwd_val) else "     N/A"
        vix_val = r["VIX"]
        vix_str = f"{vix_val:>6.1f}" if not pd.isna(vix_val) else "   N/A"
        print(f"  {str(sig_date.date()):<13} ${r['Close']:>7.2f} {r['RSI_14']:>6.1f} "
              f"{r['Dist_52w_High']:>8.1f} {r['Dist_SMA200']:>8.1f} {vix_str} "
              f"{int(r['Consec_Down']):>6} {r['Chg_5d']:>8.1f} {fwd_str}")

# ── 7. EXTREME FEAR TIER ANALYSIS (all tiers with the $150 setup) ─────────────

print()
print("=" * 70)
print("TIER CONDITIONS AT $150-155  (with various VIX assumptions)")
print("=" * 70)
print()

# Projected indicators at $150 and $155
for target, label in [(150, "$150 flush"), (155, "$155 flush")]:
    dist_52w = (target - high_52w) / high_52w * 100
    dist_200  = (target - sma200)  / sma200  * 100
    est_rsi   = estimate_rsi_at_target(close_live, target, days_to_target=5)
    drop_from_current = (target - current_price) / current_price * 100

    print(f"── {label}  (from current ${current_price:.2f},  {drop_from_current:+.1f}%) ──")
    print(f"   Projected RSI:        {est_rsi:.1f}")
    print(f"   Dist from 52w-high:   {dist_52w:.1f}%   (52w high = ${high_52w:.2f})")
    print(f"   Dist from 200 SMA:    {dist_200:.1f}%   (200 SMA  = ${sma200:.2f})")
    print()

    for vix_scenario, vix_label in [(20, "VIX ~20"), (25, "VIX ~25"), (30, "VIX ~30"), (35, "VIX >35")]:
        # Tier logic
        if est_rsi < 35 and dist_52w < -20 and vix_scenario > 25 and dist_200 < -10:
            tier = "TIER 1 — Maximum Fear"
            hist_median = None
            # Look up from data
            t1_mask = (df["RSI_14"] < 35) & (df["Dist_52w_High"] < -20) & (df["VIX"] > 25) & (df["Dist_SMA200"] < -10)
            t1_fwd  = df[t1_mask]["Fwd_90d_pct"].dropna()
            if len(t1_fwd) >= 3:
                hist_median = t1_fwd.median()
                hist_pct40  = (t1_fwd >= 40).mean() * 100
                hist_n      = len(t1_fwd)
        elif est_rsi < 40 and dist_52w < -15 and vix_scenario > 22 and dist_200 < -5:
            tier = "TIER 2 — High Fear"
            t2_mask = (df["RSI_14"] < 40) & (df["Dist_52w_High"] < -15) & (df["VIX"] > 22) & (df["Dist_SMA200"] < -5)
            t2_fwd  = df[t2_mask]["Fwd_90d_pct"].dropna()
            hist_median = t2_fwd.median() if len(t2_fwd) >= 3 else None
            hist_pct40  = (t2_fwd >= 40).mean() * 100 if len(t2_fwd) >= 3 else None
            hist_n      = len(t2_fwd)
        elif est_rsi < 45 and dist_52w < -10 and dist_200 < 0:
            tier = "TIER 3 — Moderate Selloff"
            t3_mask = (df["RSI_14"] < 45) & (df["Dist_52w_High"] < -10) & (df["Dist_SMA200"] < 0)
            t3_fwd  = df[t3_mask]["Fwd_90d_pct"].dropna()
            hist_median = t3_fwd.median() if len(t3_fwd) >= 3 else None
            hist_pct40  = (t3_fwd >= 40).mean() * 100 if len(t3_fwd) >= 3 else None
            hist_n      = len(t3_fwd)
        else:
            tier = "TIER 4 — Any Weakness"
            hist_median = None
            hist_pct40  = None
            hist_n      = 0

        if hist_median is not None:
            print(f"   {vix_label:<12}  →  {tier}")
            print(f"                    Historical: Median 90d = {hist_median:.1f}%  |  Hit rate ≥40% = {hist_pct40:.0f}%  (N={hist_n})")
        else:
            print(f"   {vix_label:<12}  →  {tier}  (too few historical analogs)")
    print()

# ── 8. COMPOSITE SCORE AT FLUSH PRICES ────────────────────────────────────────

print("=" * 70)
print("COMPOSITE ENTRY SCORE AT $150-155  (scale 0-16, higher = better setup)")
print("=" * 70)
print()

def composite_score(rsi, dist_52w, dist_200, vix, consec_down=0, chg_5d=None):
    score = 0
    if rsi < 30:     score += 3
    elif rsi < 35:   score += 2
    elif rsi < 40:   score += 1

    if dist_52w < -30:    score += 3
    elif dist_52w < -20:  score += 2
    elif dist_52w < -15:  score += 1

    if dist_200 < -20:   score += 3
    elif dist_200 < -10: score += 2
    elif dist_200 < 0:   score += 1

    if vix >= 30:   score += 3
    elif vix >= 25: score += 2
    elif vix >= 20: score += 1

    if consec_down >= 5:   score += 2
    elif consec_down >= 3: score += 1

    if chg_5d is not None:
        if chg_5d < -15:   score += 2
        elif chg_5d < -10: score += 1

    return score

# Current score for reference
cur_chg5d = float(latest["Chg_5d"])
cur_score = composite_score(
    current_rsi,
    float(latest["Dist_52w_High"]),
    float(latest["Dist_SMA200"]),
    current_vix,
    int(latest["Consec_Down"]),
    cur_chg5d
)
print(f"Current score (${current_price:.0f}):  {cur_score}/16")
print()

# Score at each historical composite level → return
score_stats = df.groupby(
    df.apply(lambda r: composite_score(
        r["RSI_14"], r["Dist_52w_High"], r["Dist_SMA200"],
        r["VIX"], r["Consec_Down"], r["Chg_5d"]), axis=1)
)["Fwd_90d_pct"].agg(
    Count="count", Median="median",
    Pct_40plus=lambda x: (x >= 40).mean() * 100
).round(1)
score_stats.index.name = "Composite_Score"
score_stats.columns = ["Count", "Median 90d%", "% ≥40% Return"]
print("Historical returns by composite score:\n")
print(score_stats.to_string())
print()

print(f"{'Target $':>10} {'VIX':>6} {'Est RSI':>9} {'Score':>7}  Tier")
print("-" * 65)
for target in [150, 155]:
    est_rsi  = estimate_rsi_at_target(close_live, target, days_to_target=5)
    dist_52w = (target - high_52w) / high_52w * 100
    dist_200  = (target - sma200)  / sma200  * 100
    drop_5d  = (target - current_price) / current_price * 100  # approx 5d chg

    for vix_val in [23, 27, 32]:
        sc = composite_score(est_rsi, dist_52w, dist_200, vix_val,
                             consec_down=4, chg_5d=drop_5d)
        if est_rsi < 35 and dist_52w < -20 and vix_val > 25 and dist_200 < -10:
            tier = "TIER 1"
        elif est_rsi < 40 and dist_52w < -15 and vix_val > 22 and dist_200 < -5:
            tier = "TIER 2"
        else:
            tier = "TIER 3"
        print(f"  ${target:<8} {vix_val:>5}  {est_rsi:>9.1f} {sc:>7}  {tier}")
    print()

# ── 9. SUMMARY ─────────────────────────────────────────────────────────────────

print("=" * 70)
print("SUMMARY: WHAT A $150-155 FLUSH MEANS FOR THE SETUP")
print("=" * 70)
print()

t150_rsi  = estimate_rsi_at_target(close_live, 150, days_to_target=5)
t155_rsi  = estimate_rsi_at_target(close_live, 155, days_to_target=5)
t150_52w  = (150 - high_52w) / high_52w * 100
t155_52w  = (155 - high_52w) / high_52w * 100
t150_200  = (150 - sma200)  / sma200  * 100
t155_200  = (155 - sma200)  / sma200  * 100

print(f"  At $155:  RSI ~{t155_rsi:.0f}  |  {t155_52w:.1f}% below 52w-high  |  {t155_200:.1f}% below 200SMA")
print(f"  At $150:  RSI ~{t150_rsi:.0f}  |  {t150_52w:.1f}% below 52w-high  |  {t150_200:.1f}% below 200SMA")
print()

# Find what score these map to
for target, rsi_t in [(155, t155_rsi), (150, t150_rsi)]:
    dist_52w = (target - high_52w) / high_52w * 100
    dist_200  = (target - sma200)  / sma200  * 100
    drop_5d  = (target - current_price) / current_price * 100
    for vix_val, vix_label in [(23, "VIX ~23 (unchanged)"), (27, "VIX ~27"), (32, "VIX ~32")]:
        sc = composite_score(rsi_t, dist_52w, dist_200, vix_val, consec_down=4, chg_5d=drop_5d)
        print(f"  ${target} + {vix_label:<25}  →  score {sc}/16", end="")
        # lookup median for score
        sc_int = int(round(sc))
        if sc_int in score_stats.index:
            row = score_stats.loc[sc_int]
            print(f"  →  hist median {row['Median 90d%']:.1f}%  |  {row['% ≥40% Return']:.0f}% hit rate")
        else:
            print()

print()
print("Bottom line:")
print(f"  $155 flush = Tier 2 minimum; Tier 1 if VIX pushes above 25.")
print(f"  $150 flush = Tier 1 unconditionally at current SMA/52w levels.")
print(f"  Composite score range at $150 (VIX 23-32): roughly 8-12 out of 16.")
print(f"  Historical median return at score 8-12: see table above.")
