"""
NVDA LEAPS Entry Condition Analysis
====================================
Finds the optimal technical conditions for buying NVDA LEAPS (12-month expiry,
targeting 40%+ gain within 90 days from fear-driven selloffs).

Period: Jan 1 2023 – present (AI era only)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys

# ── 1. FETCH DATA ─────────────────────────────────────────────────────────────

print("=" * 70)
print("NVDA LEAPS ENTRY CONDITION ANALYSIS  |  AI Era: 2023-present")
print("=" * 70)
print()

START = "2023-01-01"
END   = datetime.today().strftime("%Y-%m-%d")

print(f"Fetching NVDA price data  {START} → {END} ...")
nvda = yf.download("NVDA", start=START, end=END, auto_adjust=True, progress=False)

print(f"Fetching VIX data         {START} → {END} ...")
vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)

print(f"  NVDA rows: {len(nvda)}  |  VIX rows: {len(vix_raw)}")
print()

# Flatten multi-index if present
if isinstance(nvda.columns, pd.MultiIndex):
    nvda.columns = nvda.columns.get_level_values(0)
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)

nvda.index = pd.to_datetime(nvda.index)
vix_raw.index = pd.to_datetime(vix_raw.index)

# ── 2. TECHNICAL INDICATORS ───────────────────────────────────────────────────

df = nvda[["Open", "High", "Low", "Close", "Volume"]].copy()
df.sort_index(inplace=True)

close = df["Close"]
high  = df["High"]
low   = df["Low"]
vol   = df["Volume"]

# ── RSI (14-day) ──
def calc_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI_14"] = calc_rsi(close, 14)

# ── MACD ──
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df["MACD"]        = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# ── 200-day SMA & distance ──
df["SMA200"]       = close.rolling(200).mean()
df["Dist_SMA200"]  = (close - df["SMA200"]) / df["SMA200"] * 100  # % above/below

# ── 52-week high & distance ──
df["High_52w"]     = close.rolling(252).max()
df["Dist_52w_High"] = (close - df["High_52w"]) / df["High_52w"] * 100  # % below (negative)

# ── Volume vs 20-day average ──
df["Vol_MA20"]    = vol.rolling(20).mean()
df["Vol_Ratio"]   = vol / df["Vol_MA20"]   # >1 = above average

# ── Rate of decline: % change over past 5, 10, 20 days ──
df["Chg_5d"]  = close.pct_change(5)  * 100
df["Chg_10d"] = close.pct_change(10) * 100
df["Chg_20d"] = close.pct_change(20) * 100

# ── Consecutive down days ──
df["Daily_Ret"] = close.pct_change()
def consec_down(series):
    result = np.zeros(len(series), dtype=int)
    count  = 0
    for i, v in enumerate(series):
        if pd.isna(v):
            count = 0
        elif v < 0:
            count += 1
        else:
            count = 0
        result[i] = count
    return result

df["Consec_Down"] = consec_down(df["Daily_Ret"].values)

# ── VIX ──
vix_close = vix_raw["Close"].rename("VIX")
df = df.join(vix_close, how="left")
df["VIX"] = df["VIX"].ffill()

# ── Month (for seasonality) ──
df["Month"] = df.index.month
df["Quarter"] = df.index.quarter

# ── 90-day forward return ──
df["Fwd_90d"] = close.shift(-90) / close - 1
df["Fwd_90d_pct"] = df["Fwd_90d"] * 100

# ── Drop rows without all needed indicators ──
df.dropna(subset=["RSI_14", "SMA200", "High_52w", "MACD_Signal", "Fwd_90d"], inplace=True)

print(f"Usable rows after computing all indicators: {len(df)}")
print()

# ── 3. DEFINE CONDITION BUCKETS ───────────────────────────────────────────────

# RSI buckets
df["RSI_Bucket"] = pd.cut(df["RSI_14"],
    bins=[0, 25, 30, 35, 40, 45, 50, 100],
    labels=["RSI<25", "RSI 25-30", "RSI 30-35", "RSI 35-40", "RSI 40-45", "RSI 45-50", "RSI>50"])

# Distance from 52-week high
df["Dist52w_Bucket"] = pd.cut(df["Dist_52w_High"],
    bins=[-100, -50, -40, -30, -20, -10, 0],
    labels=[">50% below", "40-50% below", "30-40% below", "20-30% below", "10-20% below", "<10% below"])

# Distance from 200-day SMA
df["Dist200_Bucket"] = pd.cut(df["Dist_SMA200"],
    bins=[-100, -30, -20, -10, 0, 10, 20, 200],
    labels=[">30% below", "20-30% below", "10-20% below", "0-10% below", "0-10% above", "10-20% above", ">20% above"])

# Volume ratio
df["Vol_Bucket"] = pd.cut(df["Vol_Ratio"],
    bins=[0, 0.75, 1.0, 1.5, 2.0, 3.0, 100],
    labels=["Low (<0.75x)", "Normal (0.75-1x)", "Moderate (1-1.5x)", "High (1.5-2x)", "Very High (2-3x)", "Extreme (>3x)"])

# MACD histogram direction
df["MACD_Cond"] = np.where(df["MACD_Hist"] > 0, "MACD_Hist_Pos",
                  np.where(df["MACD"] < df["MACD_Signal"], "MACD_Below_Signal", "MACD_Above_Signal"))

# VIX buckets
df["VIX_Bucket"] = pd.cut(df["VIX"],
    bins=[0, 15, 20, 25, 30, 35, 40, 200],
    labels=["VIX<15", "VIX 15-20", "VIX 20-25", "VIX 25-30", "VIX 30-35", "VIX 35-40", "VIX>40"])

# Consecutive down days
df["Consec_Bucket"] = pd.cut(df["Consec_Down"],
    bins=[-1, 1, 2, 3, 4, 5, 100],
    labels=["0-1 days", "2 days", "3 days", "4 days", "5 days", "6+ days"])

# 5-day decline
df["Chg5d_Bucket"] = pd.cut(df["Chg_5d"],
    bins=[-100, -20, -15, -10, -5, 0, 5, 100],
    labels=[">20% drop", "15-20% drop", "10-15% drop", "5-10% drop", "0-5% drop", "0-5% gain", ">5% gain"])

# Month (seasonality)
month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
df["Month_Name"] = df["Month"].map(month_map)

# ── 4. SINGLE-CONDITION ANALYSIS ─────────────────────────────────────────────

def analyze_condition(df, col, min_count=5):
    """For each unique value of col, compute stats on forward 90d return."""
    rows = []
    for val, grp in df.groupby(col, observed=True):
        if len(grp) < min_count:
            continue
        fwd = grp["Fwd_90d_pct"].dropna()
        rows.append({
            "Condition": f"{col}={val}",
            "Count": len(fwd),
            "Median_Fwd90": round(fwd.median(), 1),
            "Mean_Fwd90":   round(fwd.mean(), 1),
            "Pct_Pos":      round((fwd > 0).mean() * 100, 0),
            "Pct_40plus":   round((fwd >= 40).mean() * 100, 0),
            "Min_Fwd90":    round(fwd.min(), 1),
            "Max_Fwd90":    round(fwd.max(), 1),
        })
    return pd.DataFrame(rows)

condition_cols = [
    "RSI_Bucket", "Dist52w_Bucket", "Dist200_Bucket",
    "Vol_Bucket", "VIX_Bucket", "Consec_Bucket",
    "Chg5d_Bucket", "Month_Name"
]

single_results = pd.concat(
    [analyze_condition(df, c) for c in condition_cols],
    ignore_index=True
)
single_results.sort_values("Median_Fwd90", ascending=False, inplace=True)

# ── 5. MULTI-CONDITION COMBINATION ANALYSIS ───────────────────────────────────
# Focus: RSI + Dist from 52w high + Dist from 200SMA + VIX (most important factors)

print("Running multi-condition combination analysis...")
print()

combo_rows = []
key_combos = [
    ("RSI_Bucket",    "Dist52w_Bucket"),
    ("RSI_Bucket",    "Dist200_Bucket"),
    ("RSI_Bucket",    "VIX_Bucket"),
    ("Dist52w_Bucket","Dist200_Bucket"),
    ("Dist52w_Bucket","VIX_Bucket"),
    ("Consec_Bucket", "RSI_Bucket"),
    ("Chg5d_Bucket",  "RSI_Bucket"),
    ("RSI_Bucket",    "Vol_Bucket"),
    ("RSI_Bucket",    "Month_Name"),
]

for c1, c2 in key_combos:
    for (v1, v2), grp in df.groupby([c1, c2], observed=True):
        fwd = grp["Fwd_90d_pct"].dropna()
        if len(fwd) < 5:
            continue
        combo_rows.append({
            "Cond1": f"{c1}={v1}",
            "Cond2": f"{c2}={v2}",
            "Combination": f"{v1}  +  {v2}",
            "Count": len(fwd),
            "Median_Fwd90": round(fwd.median(), 1),
            "Mean_Fwd90":   round(fwd.mean(), 1),
            "Pct_40plus":   round((fwd >= 40).mean() * 100, 0),
            "Pct_Pos":      round((fwd > 0).mean() * 100, 0),
            "Med_RSI":      round(grp["RSI_14"].median(), 1),
            "Med_VIX":      round(grp["VIX"].median(), 1),
            "Med_Dist52w":  round(grp["Dist_52w_High"].median(), 1),
            "Med_Dist200":  round(grp["Dist_SMA200"].median(), 1),
        })

combo_df = pd.DataFrame(combo_rows).sort_values("Median_Fwd90", ascending=False)

# Triple combinations on best single conditions
print("Running triple-condition analysis...")
triple_rows = []
for (v1, v2, v3), grp in df.groupby(
        ["RSI_Bucket", "Dist52w_Bucket", "Dist200_Bucket"], observed=True):
    fwd = grp["Fwd_90d_pct"].dropna()
    if len(fwd) < 5:
        continue
    triple_rows.append({
        "RSI":        str(v1),
        "Dist_52w":   str(v2),
        "Dist_200MA": str(v3),
        "Count":      len(fwd),
        "Median_Fwd90": round(fwd.median(), 1),
        "Mean_Fwd90":   round(fwd.mean(), 1),
        "Pct_40plus":   round((fwd >= 40).mean() * 100, 0),
        "Pct_Pos":      round((fwd > 0).mean() * 100, 0),
        "Med_VIX":      round(grp["VIX"].median(), 1),
        "Med_Consec":   round(grp["Consec_Down"].median(), 1),
        "Med_Chg5d":    round(grp["Chg_5d"].median(), 1),
    })

triple_df = pd.DataFrame(triple_rows).sort_values("Median_Fwd90", ascending=False)

# ── 6. BEST SINGLE-CONDITION FINDINGS ────────────────────────────────────────

print("=" * 70)
print("PART 1: SINGLE-CONDITION RANKINGS  (sorted by Median 90d Return)")
print("=" * 70)
print()

# Print top and bottom 15 single conditions
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.1f}".format)

top_single = single_results.head(20).copy()
print("Top 20 single conditions by median 90d forward return:\n")
print(f"{'Condition':<35} {'Count':>6} {'Med%':>7} {'Mean%':>7} {'%Pos':>6} {'%≥40%':>7} {'Min%':>7} {'Max%':>7}")
print("-" * 83)
for _, r in top_single.iterrows():
    print(f"{r['Condition']:<35} {r['Count']:>6} {r['Median_Fwd90']:>7.1f} {r['Mean_Fwd90']:>7.1f} "
          f"{r['Pct_Pos']:>6.0f} {r['Pct_40plus']:>7.0f} {r['Min_Fwd90']:>7.1f} {r['Max_Fwd90']:>7.1f}")

# ── 7. BEST COMBO FINDINGS ────────────────────────────────────────────────────

print()
print("=" * 70)
print("PART 2: TWO-CONDITION COMBINATIONS  (sorted by Median 90d Return)")
print("=" * 70)
print()

top_combos = combo_df.head(25).copy()
print(f"{'Combination':<50} {'N':>4} {'Med%':>7} {'Mean%':>7} {'%≥40%':>7} {'%Pos':>6} {'VIX':>6} {'RSI':>6} {'Δ52w%':>7} {'Δ200%':>7}")
print("-" * 110)
for _, r in top_combos.iterrows():
    print(f"{r['Combination']:<50} {r['Count']:>4} {r['Median_Fwd90']:>7.1f} {r['Mean_Fwd90']:>7.1f} "
          f"{r['Pct_40plus']:>7.0f} {r['Pct_Pos']:>6.0f} {r['Med_VIX']:>6.1f} {r['Med_RSI']:>6.1f} "
          f"{r['Med_Dist52w']:>7.1f} {r['Med_Dist200']:>7.1f}")

# ── 8. TRIPLE CONDITION ANALYSIS ─────────────────────────────────────────────

print()
print("=" * 70)
print("PART 3: RSI × DIST-FROM-52W-HIGH × DIST-FROM-200MA  (triple combos)")
print("=" * 70)
print()

top_triple = triple_df.head(20).copy()
print(f"{'RSI':<12} {'Dist-52w':<18} {'Dist-200MA':<18} {'N':>4} {'Med%':>7} {'Mean%':>7} {'%≥40%':>7} {'%Pos':>6} {'VIX':>6} {'Cons↓':>6} {'Chg5d%':>8}")
print("-" * 110)
for _, r in top_triple.iterrows():
    print(f"{r['RSI']:<12} {r['Dist_52w']:<18} {r['Dist_200MA']:<18} {r['Count']:>4} "
          f"{r['Median_Fwd90']:>7.1f} {r['Mean_Fwd90']:>7.1f} {r['Pct_40plus']:>7.0f} "
          f"{r['Pct_Pos']:>6.0f} {r['Med_VIX']:>6.1f} {r['Med_Consec']:>6.1f} {r['Med_Chg5d']:>8.1f}")

# ── 9. FIND ACTUAL DATES OF OPTIMAL ENTRIES ───────────────────────────────────

print()
print("=" * 70)
print("PART 4: ACTUAL HISTORICAL ENTRY OPPORTUNITIES")
print("=" * 70)
print()

# Define "optimal" as: RSI<40, >15% below 52w high, below 200MA
optimal_mask = (
    (df["RSI_14"] < 40) &
    (df["Dist_52w_High"] < -15) &
    (df["Dist_SMA200"] < 0)
)
optimal_entries = df[optimal_mask].copy()

print(f"Days meeting: RSI<40  AND  >15% below 52w-high  AND  below 200SMA: {len(optimal_entries)}")
print()

# Cluster consecutive days (only show first day of each cluster as the entry signal)
if len(optimal_entries) > 0:
    dates = optimal_entries.index
    clusters = []
    cluster_start = dates[0]
    prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days > 5:
            clusters.append(cluster_start)
            cluster_start = d
        prev = d
    clusters.append(cluster_start)

    print(f"Distinct entry clusters (signal dates): {len(clusters)}")
    years_covered = (df.index[-1] - df.index[0]).days / 365.25
    print(f"Data covers {years_covered:.1f} years  →  ~{len(clusters)/years_covered:.1f} opportunities/year")
    print()

    print(f"{'Date':<13} {'Close':>8} {'RSI':>6} {'Δ52w%':>8} {'Δ200%':>8} {'VIX':>6} {'Cons↓':>6} {'Chg5d%':>8} {'Fwd90d%':>9}")
    print("-" * 80)

    total_opp_count = 0
    for sig_date in clusters:
        r = df.loc[sig_date]
        fwd = r["Fwd_90d_pct"]
        fwd_str = f"{fwd:>8.1f}" if not pd.isna(fwd) else "     N/A"
        vix = r["VIX"] if not pd.isna(r["VIX"]) else float("nan")
        vix_str = f"{vix:>6.1f}" if not pd.isna(vix) else "   N/A"
        print(f"{str(sig_date.date()):<13} ${r['Close']:>7.2f} {r['RSI_14']:>6.1f} "
              f"{r['Dist_52w_High']:>8.1f} {r['Dist_SMA200']:>8.1f} {vix_str} "
              f"{int(r['Consec_Down']):>6} {r['Chg_5d']:>8.1f} {fwd_str}")
        total_opp_count += 1

# ── 10. SEASONALITY ANALYSIS ─────────────────────────────────────────────────

print()
print("=" * 70)
print("PART 5: SEASONALITY — Monthly Return Distribution")
print("=" * 70)
print()

month_stats = df.groupby("Month_Name")["Fwd_90d_pct"].agg(
    Count="count", Median="median", Mean="mean",
    Pct_40plus=lambda x: (x >= 40).mean() * 100
).round(1)

# Re-order by calendar month
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_stats = month_stats.reindex([m for m in month_order if m in month_stats.index])
month_stats.columns = ["Count", "Median Fwd90%", "Mean Fwd90%", "% ≥40% Return"]
print(month_stats.to_string())

# ── 11. VIX CORRELATION ───────────────────────────────────────────────────────

print()
print("=" * 70)
print("PART 6: VIX LEVEL AT ENTRY vs. 90d Forward Return")
print("=" * 70)
print()

vix_stats = df.groupby("VIX_Bucket", observed=True)["Fwd_90d_pct"].agg(
    Count="count", Median="median", Mean="mean",
    Pct_40plus=lambda x: (x >= 40).mean() * 100
).round(1)
vix_stats.columns = ["Count", "Median Fwd90%", "Mean Fwd90%", "% ≥40% Return"]
print(vix_stats.to_string())

# ── 12. CONSECUTIVE DOWN DAYS ANALYSIS ───────────────────────────────────────

print()
print("=" * 70)
print("PART 7: CONSECUTIVE DOWN DAYS BEFORE ENTRY vs. 90d Forward Return")
print("=" * 70)
print()

consec_stats = df.groupby("Consec_Bucket", observed=True)["Fwd_90d_pct"].agg(
    Count="count", Median="median", Mean="mean",
    Pct_40plus=lambda x: (x >= 40).mean() * 100
).round(1)
consec_stats.columns = ["Count", "Median Fwd90%", "Mean Fwd90%", "% ≥40% Return"]
print(consec_stats.to_string())

# ── 13. COMPOSITE SCORE / FINAL SUMMARY ──────────────────────────────────────

print()
print("=" * 70)
print("PART 8: OPTIMAL ENTRY SCORECARD SUMMARY")
print("=" * 70)
print()

# Based on all analysis, create a composite score and define the optimal entry window
def get_score(row):
    score = 0
    # RSI
    if row["RSI_14"] < 30:     score += 3
    elif row["RSI_14"] < 35:   score += 2
    elif row["RSI_14"] < 40:   score += 1

    # Distance from 52w high
    if row["Dist_52w_High"] < -30:    score += 3
    elif row["Dist_52w_High"] < -20:  score += 2
    elif row["Dist_52w_High"] < -15:  score += 1

    # Distance from 200SMA
    if row["Dist_SMA200"] < -20:   score += 3
    elif row["Dist_SMA200"] < -10: score += 2
    elif row["Dist_SMA200"] < 0:   score += 1

    # VIX
    if row["VIX"] >= 30:   score += 3
    elif row["VIX"] >= 25: score += 2
    elif row["VIX"] >= 20: score += 1

    # Consecutive down days
    if row["Consec_Down"] >= 5:   score += 2
    elif row["Consec_Down"] >= 3: score += 1

    # Recent drop
    if row["Chg_5d"] < -15:    score += 2
    elif row["Chg_5d"] < -10:  score += 1

    return score

df["Composite_Score"] = df.apply(get_score, axis=1)

score_stats = df.groupby("Composite_Score")["Fwd_90d_pct"].agg(
    Count="count", Median="median", Mean="mean",
    Pct_40plus=lambda x: (x >= 40).mean() * 100
).round(1)
score_stats.columns = ["Count", "Median Fwd90%", "Mean Fwd90%", "% ≥40% Return"]
print("Composite Score (0-16): higher = more extreme fear/selloff conditions")
print()
print(score_stats.to_string())

# High-score entries
print()
print("High-scoring entry days (score ≥ 8):")
high_score = df[df["Composite_Score"] >= 8].sort_values("Composite_Score", ascending=False)
print(f"\n{'Date':<13} {'Score':>6} {'Close':>8} {'RSI':>6} {'Δ52w%':>8} {'Δ200%':>8} {'VIX':>6} {'Cons↓':>6} {'Chg5d%':>8} {'Fwd90d%':>9}")
print("-" * 85)
for idx, r in high_score.iterrows():
    fwd = r["Fwd_90d_pct"]
    fwd_str = f"{fwd:>8.1f}" if not pd.isna(fwd) else "     N/A"
    vix = r["VIX"]
    vix_str = f"{vix:>6.1f}" if not pd.isna(vix) else "   N/A"
    print(f"{str(idx.date()):<13} {int(r['Composite_Score']):>6} ${r['Close']:>7.2f} "
          f"{r['RSI_14']:>6.1f} {r['Dist_52w_High']:>8.1f} {r['Dist_SMA200']:>8.1f} "
          f"{vix_str} {int(r['Consec_Down']):>6} {r['Chg_5d']:>8.1f} {fwd_str}")

# ── 14. FINAL RECOMMENDATION TABLE ───────────────────────────────────────────

print()
print("=" * 70)
print("PART 9: FINAL — CONDITIONS RANKED BY 90d RETURN QUALITY")
print("         (patterns appearing ≥5 times only)")
print("=" * 70)
print()

# Build the definitive condition importance table
summary_rows = []
checks = [
    ("RSI < 25",          df["RSI_14"] < 25),
    ("RSI 25-30",         (df["RSI_14"] >= 25) & (df["RSI_14"] < 30)),
    ("RSI 30-35",         (df["RSI_14"] >= 30) & (df["RSI_14"] < 35)),
    ("RSI 35-40",         (df["RSI_14"] >= 35) & (df["RSI_14"] < 40)),
    ("RSI 40-45",         (df["RSI_14"] >= 40) & (df["RSI_14"] < 45)),
    (">30% below 52w-high", df["Dist_52w_High"] < -30),
    ("20-30% below 52w-high", (df["Dist_52w_High"] >= -30) & (df["Dist_52w_High"] < -20)),
    ("10-20% below 52w-high", (df["Dist_52w_High"] >= -20) & (df["Dist_52w_High"] < -10)),
    (">20% below 200SMA",   df["Dist_SMA200"] < -20),
    ("10-20% below 200SMA", (df["Dist_SMA200"] >= -20) & (df["Dist_SMA200"] < -10)),
    ("0-10% below 200SMA",  (df["Dist_SMA200"] >= -10) & (df["Dist_SMA200"] < 0)),
    ("VIX > 35",           df["VIX"] > 35),
    ("VIX 30-35",          (df["VIX"] >= 30) & (df["VIX"] < 35)),
    ("VIX 25-30",          (df["VIX"] >= 25) & (df["VIX"] < 30)),
    ("VIX 20-25",          (df["VIX"] >= 20) & (df["VIX"] < 25)),
    ("VIX < 20",           df["VIX"] < 20),
    ("≥5 consec down days", df["Consec_Down"] >= 5),
    ("3-4 consec down days", (df["Consec_Down"] >= 3) & (df["Consec_Down"] < 5)),
    ("5d drop > 15%",      df["Chg_5d"] < -15),
    ("5d drop 10-15%",     (df["Chg_5d"] >= -15) & (df["Chg_5d"] < -10)),
    ("5d drop 5-10%",      (df["Chg_5d"] >= -10) & (df["Chg_5d"] < -5)),
    ("High volume (>2x avg)", df["Vol_Ratio"] > 2),
    ("Volume 1.5-2x avg",  (df["Vol_Ratio"] >= 1.5) & (df["Vol_Ratio"] < 2)),
]

for label, mask in checks:
    sub = df[mask]["Fwd_90d_pct"].dropna()
    if len(sub) < 5:
        continue
    summary_rows.append({
        "Condition": label,
        "Occurrences": len(sub),
        "Median_90d%": round(sub.median(), 1),
        "Mean_90d%":   round(sub.mean(), 1),
        "% Days ≥40% Return": round((sub >= 40).mean() * 100, 0),
        "% Days Positive":    round((sub > 0).mean() * 100, 0),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Median_90d%", ascending=False)

print(f"{'Condition':<30} {'N':>5} {'Median%':>8} {'Mean%':>7} {'%≥40%':>7} {'%Pos':>6}")
print("-" * 65)
for _, r in summary_df.iterrows():
    print(f"{r['Condition']:<30} {r['Occurrences']:>5} {r['Median_90d%']:>8.1f} {r['Mean_90d%']:>7.1f} "
          f"{r['% Days ≥40% Return']:>7.0f} {r['% Days Positive']:>6.0f}")

# ── 15. THE OPTIMAL COMBINED ENTRY SIGNAL ─────────────────────────────────────

print()
print("=" * 70)
print("PART 10: THE OPTIMAL COMBINED ENTRY SIGNAL")
print("=" * 70)
print()

# Test a tiered entry: strict (all best conditions met) vs moderate
tiers = [
    ("TIER 1 — Maximum Fear (strictest)",
     (df["RSI_14"] < 35) & (df["Dist_52w_High"] < -20) & (df["VIX"] > 25) & (df["Dist_SMA200"] < -10)),
    ("TIER 2 — High Fear",
     (df["RSI_14"] < 40) & (df["Dist_52w_High"] < -15) & (df["VIX"] > 22) & (df["Dist_SMA200"] < -5)),
    ("TIER 3 — Moderate Selloff",
     (df["RSI_14"] < 45) & (df["Dist_52w_High"] < -10) & (df["Dist_SMA200"] < 0)),
    ("TIER 4 — Any Weakness",
     (df["RSI_14"] < 50) & (df["Dist_52w_High"] < -5)),
]

for label, mask in tiers:
    sub = df[mask]["Fwd_90d_pct"].dropna()
    days = df[mask]
    if len(sub) < 5:
        print(f"{label}: only {len(sub)} occurrences — skipping")
        continue
    # Count clusters
    sig_dates = df[mask].index
    n_clusters = 1 if len(sig_dates) > 0 else 0
    prev = sig_dates[0] if len(sig_dates) > 0 else None
    for d in sig_dates[1:]:
        if (d - prev).days > 5:
            n_clusters += 1
        prev = d
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    print(f"{label}")
    print(f"  Days matching:  {len(sub)}  |  Entry clusters: {n_clusters}  |  ~{n_clusters/yrs:.1f} per year")
    print(f"  Median 90d:  {sub.median():.1f}%   Mean: {sub.mean():.1f}%")
    print(f"  % achieving ≥40% in 90d: {(sub>=40).mean()*100:.0f}%")
    print(f"  % positive return:        {(sub>0).mean()*100:.0f}%")
    print(f"  Range: [{sub.min():.1f}%, {sub.max():.1f}%]")
    print()

# ── Save data ──────────────────────────────────────────────────────────────────
df.to_csv("nvda_indicators.csv")
summary_df.to_csv("nvda_condition_summary.csv", index=False)
triple_df.to_csv("nvda_triple_combos.csv", index=False)
print("Raw data saved to: nvda_indicators.csv")
print("Summary table saved to: nvda_condition_summary.csv")
print("Triple combos saved to: nvda_triple_combos.csv")
