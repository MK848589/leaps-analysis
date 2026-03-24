"""
NVDA BOTTOM SIGNAL ANALYSIS
============================
Identifies confirmed NVDA price bottoms from Jan 2023–present and determines
the exact indicator combination that most reliably predicted each bottom.

Indicators: MACD(12,26,9), RSI(14), 9 EMA, 48 EMA, 200 EMA
Bottom definition: Local price low followed by ≥15% recovery within 60 trading days
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.3f}'.format)

# ── 1. FETCH DATA ─────────────────────────────────────────────────────────────
print("=" * 90)
print("  NVDA LEAPS BOTTOM SIGNAL ANALYSIS  |  Jan 2023 → Today")
print("=" * 90)

START = "2023-01-01"
END   = datetime.today().strftime("%Y-%m-%d")

nvda = yf.download("NVDA", start=START, end=END, auto_adjust=True, progress=False)
if isinstance(nvda.columns, pd.MultiIndex):
    nvda.columns = nvda.columns.get_level_values(0)
nvda.index = pd.to_datetime(nvda.index)
nvda.sort_index(inplace=True)

vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)
vix_raw.index = pd.to_datetime(vix_raw.index)

df = nvda[["Open", "High", "Low", "Close", "Volume"]].copy()

# ── 2. INDICATORS ─────────────────────────────────────────────────────────────
close = df["Close"]

# EMAs
df["EMA9"]  = close.ewm(span=9,   adjust=False).mean()
df["EMA48"] = close.ewm(span=48,  adjust=False).mean()
df["EMA200"]= close.ewm(span=200, adjust=False).mean()

# MACD (12, 26, 9)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df["MACD"]        = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
df["MACD_Hist_prev"] = df["MACD_Hist"].shift(1)

# RSI (14)
delta = close.diff()
gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
df["RSI"] = 100 - (100 / (1 + gain / loss))

# VIX
df["VIX"] = vix_raw["Close"].reindex(df.index, method="ffill")

# Drawdown from rolling 52-week high
df["Roll52w_High"] = close.rolling(252, min_periods=1).max()
df["Drawdown_pct"] = (close - df["Roll52w_High"]) / df["Roll52w_High"] * 100

# EMA crossover: 9 cross 48
df["EMA9_above_48"]   = df["EMA9"] > df["EMA48"]
df["EMA9_cross_48_bull"] = (~df["EMA9_above_48"].shift(1).fillna(False)) & df["EMA9_above_48"]

# ── 3. IDENTIFY CONFIRMED BOTTOMS ─────────────────────────────────────────────
# A confirmed bottom is a local low (lowest close in 10-day window around it)
# followed by ≥15% gain within 60 trading days.
# Also require drawdown of at least 10% from recent high to filter out noise.

MIN_DRAWDOWN  = -10.0   # must be ≥10% below 52-week high
MIN_RECOVERY  = 15.0    # must recover ≥15% from bottom within 60 days
LOOKBACK_DAYS = 10      # local minimum window (5 days each side)
FORWARD_DAYS  = 60

closes = df["Close"].values
dates  = df.index
n      = len(df)

bottoms = []

for i in range(LOOKBACK_DAYS, n - FORWARD_DAYS):
    # Local minimum: lowest close in ±10 day window
    window_start = max(0, i - LOOKBACK_DAYS)
    window_end   = min(n, i + LOOKBACK_DAYS + 1)
    local_min    = closes[window_start:window_end].min()

    if closes[i] != local_min:
        continue

    # Drawdown filter
    dd = df["Drawdown_pct"].iloc[i]
    if dd > MIN_DRAWDOWN:
        continue

    # Minimum separation from other bottoms (at least 15 days)
    if bottoms and (dates[i] - bottoms[-1]["date"]).days < 15:
        continue

    # Forward recovery
    fwd_prices  = closes[i:i + FORWARD_DAYS + 1]
    fwd_max     = fwd_prices.max()
    recovery    = (fwd_max - closes[i]) / closes[i] * 100

    if recovery < MIN_RECOVERY:
        continue

    # 60-day and 90-day forward close returns
    fwd60_idx = min(i + FORWARD_DAYS, n - 1)
    fwd90_idx = min(i + 90, n - 1)
    ret60 = (closes[fwd60_idx] - closes[i]) / closes[i] * 100
    ret90 = (closes[fwd90_idx] - closes[i]) / closes[i] * 100

    # Days until recovery peak
    peak_offset = int(np.argmax(fwd_prices))

    bottoms.append({
        "idx":         i,
        "date":        dates[i],
        "close":       round(closes[i], 2),
        "drawdown":    round(dd, 1),
        "recovery_pct":round(recovery, 1),
        "peak_days":   peak_offset,
        "ret60":       round(ret60, 1),
        "ret90":       round(ret90, 1),
    })

print(f"\nFound {len(bottoms)} confirmed bottoms (≥10% drawdown → ≥15% recovery within 60 days)\n")

# ── 4. INDICATOR ANALYSIS AT EACH BOTTOM ─────────────────────────────────────

def macd_status(row):
    hist     = row["MACD_Hist"]
    hist_p   = row["MACD_Hist_prev"]
    macd_v   = row["MACD"]
    signal_v = row["MACD_Signal"]

    if pd.isna(hist_p):
        expanding = None
    else:
        expanding = abs(hist) > abs(hist_p)

    cross_bull = (macd_v > signal_v) and (macd_v - signal_v < 0.3)  # just crossed
    bull_zone  = macd_v > signal_v

    return hist, expanding, bull_zone

def ema_position(row):
    """Returns which EMAs price is below"""
    p   = row["Close"]
    e9  = row["EMA9"]
    e48 = row["EMA48"]
    e200= row["EMA200"]
    below9   = p < e9
    below48  = p < e48
    below200 = p < e200
    return below9, below48, below200, p/e9-1, p/e48-1, p/e200-1

# ── 5. FIND MACD CROSS TIMING RELATIVE TO BOTTOM ──────────────────────────────

def find_macd_cross(df, bottom_idx, window=20):
    """
    Find nearest bullish MACD crossover (MACD crosses above Signal) within ±20 days.
    Returns days offset (negative = before bottom, positive = after).
    """
    macd   = df["MACD"].values
    signal = df["MACD_Signal"].values

    for offset in range(-window, window+1):
        j = bottom_idx + offset
        if j <= 0 or j >= len(df):
            continue
        # Bullish cross: MACD > Signal today AND MACD <= Signal yesterday
        if macd[j] > signal[j] and macd[j-1] <= signal[j-1]:
            return offset
    return None

def find_ema9_cross_above_48(df, bottom_idx, window=25):
    """Find nearest bullish 9/48 EMA crossover within ±25 days. Returns day offset."""
    e9  = df["EMA9"].values
    e48 = df["EMA48"].values
    for offset in range(-window, window+1):
        j = bottom_idx + offset
        if j <= 0 or j >= len(df):
            continue
        if e9[j] > e48[j] and e9[j-1] <= e48[j-1]:
            return offset
    return None

def ema_cross_sequence(df, bottom_idx, window=30):
    """After bottom, in what order did price cross back above each EMA?"""
    closes = df["Close"].values
    e9  = df["EMA9"].values
    e48 = df["EMA48"].values
    e200= df["EMA200"].values

    cross9 = cross48 = cross200 = None

    # Was price already below each EMA at bottom?
    bi = bottom_idx
    was_below9   = closes[bi] < e9[bi]
    was_below48  = closes[bi] < e48[bi]
    was_below200 = closes[bi] < e200[bi]

    for offset in range(1, window+1):
        j = bi + offset
        if j >= len(df):
            break
        if cross9   is None and was_below9   and closes[j] > e9[j]:
            cross9   = offset
        if cross48  is None and was_below48  and closes[j] > e48[j]:
            cross48  = offset
        if cross200 is None and was_below200 and closes[j] > e200[j]:
            cross200 = offset

    return cross9, cross48, cross200

def additional_downside(df, bottom_idx, signal_day_offset):
    """
    If signal fires N days before/after the bottom,
    what was max additional downside from signal day to bottom?
    """
    signal_day_idx = bottom_idx + signal_day_offset
    if signal_day_idx < 0 or signal_day_idx >= len(df):
        return 0.0
    signal_price = df["Close"].iloc[signal_day_idx]
    bottom_price = df["Close"].iloc[bottom_idx]
    if signal_day_idx <= bottom_idx:
        # Signal before bottom — downside = bottom below signal
        return round((bottom_price - signal_price) / signal_price * 100, 2)
    else:
        return 0.0  # signal after bottom, no more downside

# ── 6. BUILD BOTTOM TABLE ─────────────────────────────────────────────────────

records = []

for b in bottoms:
    i    = b["idx"]
    row  = df.iloc[i]

    rsi  = round(row["RSI"], 1)
    hist, expanding, bull_zone = macd_status(row)
    below9, below48, below200, pct9, pct48, pct200 = ema_position(row)

    macd_cross_offset = find_macd_cross(df, i)
    ema_cross_offset  = find_ema9_cross_above_48(df, i)
    cross9_days, cross48_days, cross200_days = ema_cross_sequence(df, i)

    # ── SIGNAL SCORING ────────────────────────────────────────────────────────
    # Condition 1: RSI oversold (≤35) → 1 point
    # Condition 2: MACD histogram contracting (negative, but getting less negative)
    #              OR MACD cross within ±3 days → 1 point
    # Condition 3: Price below EMA9 AND EMA48 (beaten down) → 1 point
    # Condition 4: Price above or touching EMA200 (long-term support) OR
    #              price < 15% below EMA200 → 1 point
    # Condition 5: EMA9/48 crossover within 5 days OR MACD bull cross within 3 days → 1 point

    c1 = int(rsi <= 40)

    # MACD hist contracting = hist negative but abs(hist) shrinking
    hist_contracting = (hist < 0) and (not expanding)
    macd_cross_near  = (macd_cross_offset is not None) and (abs(macd_cross_offset) <= 5)
    c2 = int(hist_contracting or macd_cross_near or bull_zone)

    c3 = int(below9 and below48)

    # Price within 20% of EMA200 (not completely disconnected)
    c4 = int(pct200 > -20)

    ema_cross_near = (ema_cross_offset is not None) and (abs(ema_cross_offset) <= 7)
    c5 = int(macd_cross_near or ema_cross_near)

    score = c1 + c2 + c3 + c4 + c5

    if score == 5:
        signal = "CONFIRMED"
    elif score == 4:
        signal = "PROBABLE"
    elif score == 3:
        signal = "POSSIBLE"
    else:
        signal = "WEAK"

    # EMA positioning description
    ema_pos_parts = []
    if below9:   ema_pos_parts.append("< EMA9")
    if below48:  ema_pos_parts.append("< EMA48")
    if below200: ema_pos_parts.append("< EMA200")
    if not ema_pos_parts: ema_pos_parts = ["Above all"]
    ema_pos_str = ", ".join(ema_pos_parts)

    macd_desc = "Bull✓" if bull_zone else ("Contracting" if hist_contracting else "Expanding↓")
    cross_str = f"{macd_cross_offset:+d}d" if macd_cross_offset is not None else "None"
    ema_cross_str = f"{ema_cross_offset:+d}d" if ema_cross_offset is not None else "None"

    # Additional downside if entering on MACD cross day
    add_dd = 0.0
    if macd_cross_offset is not None and macd_cross_offset < 0:
        add_dd = additional_downside(df, i, macd_cross_offset)

    records.append({
        "Date":          b["date"].strftime("%Y-%m-%d"),
        "Price":         b["close"],
        "Drawdown%":     b["drawdown"],
        "RSI":           rsi,
        "MACD_Hist":     round(hist, 3),
        "MACD_Status":   macd_desc,
        "MACD_Cross":    cross_str,
        "EMA_Position":  ema_pos_str,
        "9/48_Cross":    ema_cross_str,
        "Pct_EMA200":    f"{pct200*100:+.1f}%",
        "Score":         f"{score}/5",
        "Signal":        signal,
        "ExtraDD%":      add_dd,
        "Ret_60d%":      b["ret60"],
        "Ret_90d%":      b["ret90"],
        "C1_RSI":        c1,
        "C2_MACD":       c2,
        "C3_EMA_Pos":    c3,
        "C4_EMA200":     c4,
        "C5_Cross":      c5,
    })

bdf = pd.DataFrame(records)

# ── 7. PRINT MAIN TABLE ───────────────────────────────────────────────────────

print("\n" + "─" * 90)
print("  CONFIRMED BOTTOMS — Full Indicator Breakdown")
print("─" * 90)

display_cols = ["Date","Price","Drawdown%","RSI","MACD_Status","MACD_Cross",
                "EMA_Position","9/48_Cross","Pct_EMA200","Score","Signal","ExtraDD%","Ret_60d%","Ret_90d%"]
print(bdf[display_cols].to_string(index=False))

# ── 8. CONDITION FREQUENCY ANALYSIS ──────────────────────────────────────────

print("\n" + "─" * 90)
print("  CONDITION FREQUENCY AT ALL BOTTOMS")
print("─" * 90)

total = len(bdf)
for col, label in [
    ("C1_RSI",    "C1: RSI ≤ 40"),
    ("C2_MACD",   "C2: MACD Hist contracting / cross ±5d / above signal"),
    ("C3_EMA_Pos","C3: Price < EMA9 AND EMA48"),
    ("C4_EMA200", "C4: Price within 20% of EMA200"),
    ("C5_Cross",  "C5: MACD or 9/48 EMA cross within ±5/7d"),
]:
    hits = bdf[col].sum()
    print(f"  {label:52s}  {hits}/{total}  ({hits/total*100:.0f}%)")

# ── 9. RETURN SUMMARY BY SIGNAL TYPE ─────────────────────────────────────────

print("\n" + "─" * 90)
print("  RETURNS BY SIGNAL STRENGTH")
print("─" * 90)

for sig in ["CONFIRMED", "PROBABLE", "POSSIBLE", "WEAK"]:
    sub = bdf[bdf["Signal"] == sig]
    if len(sub) == 0:
        continue
    r60  = sub["Ret_60d%"].mean()
    r90  = sub["Ret_90d%"].mean()
    add  = sub["ExtraDD%"].mean()
    print(f"  {sig:12s}  n={len(sub)}  "
          f"Avg 60d: {r60:+6.1f}%   Avg 90d: {r90:+6.1f}%   Avg ExtraDD: {add:+5.1f}%")

# ── 10. CURRENT READING ───────────────────────────────────────────────────────

print("\n" + "─" * 90)
print("  CURRENT INDICATOR READING  (Latest day)")
print("─" * 90)

last  = df.iloc[-1]
ldate = df.index[-1].strftime("%Y-%m-%d")
lclose = last["Close"]
lrsi   = last["RSI"]
lhist  = last["MACD_Hist"]
lmacd  = last["MACD"]
lsig   = last["MACD_Signal"]
le9    = last["EMA9"]
le48   = last["EMA48"]
le200  = last["EMA200"]
ldd    = last["Drawdown_pct"]

# Current score
cur_c1 = int(lrsi <= 40)
cur_hist_contracting = (lhist < 0) and (abs(lhist) < abs(df["MACD_Hist"].iloc[-2]))
cur_c2 = int(cur_hist_contracting or lmacd > lsig)
cur_c3 = int(lclose < le9 and lclose < le48)
cur_c4 = int((lclose / le200 - 1) * 100 > -20)
cur_c5 = int(cur_hist_contracting)  # conservative: no confirmed cross yet
cur_score = cur_c1 + cur_c2 + cur_c3 + cur_c4 + cur_c5

print(f"\n  Date:            {ldate}")
print(f"  Close:           ${lclose:.2f}")
print(f"  Drawdown from 52w High:  {ldd:.1f}%")
print()
print(f"  RSI (14):        {lrsi:.1f}  {'✓ OVERSOLD (≤40)' if lrsi<=40 else '✗ Not oversold'}")
print(f"  MACD:            {lmacd:.3f}")
print(f"  MACD Signal:     {lsig:.3f}")
print(f"  MACD Hist:       {lhist:.3f}  ({'Contracting' if cur_hist_contracting else 'Still expanding/bearish'})")
print(f"  EMA9:            {le9:.2f}  (Price {'below' if lclose < le9 else 'above'} by {(lclose/le9-1)*100:+.1f}%)")
print(f"  EMA48:           {le48:.2f}  (Price {'below' if lclose < le48 else 'above'} by {(lclose/le48-1)*100:+.1f}%)")
print(f"  EMA200:          {le200:.2f}  (Price {'below' if lclose < le200 else 'above'} by {(lclose/le200-1)*100:+.1f}%)")
print()
print(f"  Current Score:   {cur_score}/5")

if cur_score == 5:
    print("  Signal:          *** CONFIRMED BOTTOM ***")
elif cur_score == 4:
    print("  Signal:          ** PROBABLE BOTTOM **")
elif cur_score == 3:
    print("  Signal:          * POSSIBLE BOTTOM *")
else:
    print("  Signal:          NOT YET TRIGGERED")

print()
print(f"  Conditions met:")
print(f"    C1 RSI ≤ 40:              {'✓' if cur_c1 else '✗ (RSI = ' + str(round(lrsi,1)) + ')'}")
print(f"    C2 MACD improving:        {'✓' if cur_c2 else '✗ (Hist=' + str(round(lhist,3)) + ', still expanding)'}")
print(f"    C3 Price < EMA9 & EMA48:  {'✓' if cur_c3 else '✗'}")
print(f"    C4 Within 20% EMA200:     {'✓' if cur_c4 else '✗'}")
print(f"    C5 Cross signal forming:  {'✓' if cur_c5 else '✗ (no confirmed cross yet)'}")

# ── 11. TRIGGER LEVELS FOR CONFIRMED BOTTOM ───────────────────────────────────

print("\n" + "─" * 90)
print("  WHAT WOULD TRIGGER CONFIRMED BOTTOM FROM HERE")
print("─" * 90)

# C1: RSI needs to be ≤ 40
# C2: MACD hist needs to start contracting (shrinking toward 0)
# C3: Already likely met if price stays down
# C4: Already met
# C5: MACD needs to cross signal line OR 9/48 cross

print(f"""
  Missing conditions and what would trigger them:

  C1 RSI ≤ 40:   RSI is currently {lrsi:.1f}.
                 {'ALREADY MET' if cur_c1 else 'Needs ~2-4 more down days OR a consolidation week at current levels.'}

  C2 MACD Hist:  Currently {lhist:.3f} and {'CONTRACTING ✓' if cur_hist_contracting else 'still expanding (getting more negative).'}
                 {'ALREADY MET' if cur_c2 else 'Needs 1 green day or pause in selling → hist stops growing.'}

  C3 EMA Pos:    Price ${lclose:.2f} vs EMA9 ${le9:.2f} / EMA48 ${le48:.2f}.
                 {'ALREADY MET' if cur_c3 else 'Not triggered.'}

  C4 EMA200:     Price is {(lclose/le200-1)*100:+.1f}% from EMA200 (${le200:.2f}).
                 {'ALREADY MET' if cur_c4 else 'Need to recover to >' + str(round(le200 * 0.80, 2)) + '.'}

  C5 Cross:      Need MACD line to cross above Signal OR EMA9 cross above EMA48.
                 MACD cross would need MACD ({lmacd:.3f}) to rise above Signal ({lsig:.3f}).
                 Gap to close: {abs(lmacd - lsig):.3f} → likely {'3-7' if abs(lmacd - lsig) > 1 else '1-3'} trading days of recovery.
""")

# ── 12. AVERAGE ENTRY TIMING STATS ────────────────────────────────────────────

print("─" * 90)
print("  ENTRY TIMING: How many days after signal did price dip further?")
print("─" * 90)

confirmed_only = bdf[bdf["Signal"].isin(["CONFIRMED", "PROBABLE"])]
if len(confirmed_only) > 0:
    avg_extra_dd = confirmed_only["ExtraDD%"].mean()
    max_extra_dd = confirmed_only["ExtraDD%"].min()  # most negative
    avg_r60 = confirmed_only["Ret_60d%"].mean()
    avg_r90 = confirmed_only["Ret_90d%"].mean()

    print(f"\n  Confirmed + Probable signals combined (n={len(confirmed_only)})")
    print(f"  Average additional downside after signal:  {avg_extra_dd:+.1f}%")
    print(f"  Worst additional downside (max pain):      {max_extra_dd:+.1f}%")
    print(f"  Average 60-day return from signal:         {avg_r60:+.1f}%")
    print(f"  Average 90-day return from signal:         {avg_r90:+.1f}%")

# Also show average days to bottom after MACD cross fires early
macd_early = [r for r in records if r["MACD_Cross"] not in ["None"] and
              r["MACD_Cross"].startswith("-")]
if macd_early:
    offsets = [int(r["MACD_Cross"].replace("d","")) for r in macd_early]
    print(f"\n  When MACD cross fired BEFORE price bottom (n={len(macd_early)}):")
    print(f"  Avg days early: {abs(np.mean(offsets)):.1f}  Max early: {abs(min(offsets))} days")
    print(f"  (This means buying on the MACD cross still left ~{abs(np.mean(offsets)):.0f} days of potential further pain)")

print("\n" + "─" * 90)
print("  BOTTOM INDICATOR PATTERN SUMMARY (What appeared at every major bottom)")
print("─" * 90)

# Compute medians for confirmed bottoms
conf = bdf[bdf["Signal"].isin(["CONFIRMED", "PROBABLE"])]
if len(conf) > 0:
    print(f"""
  Pattern observed across all Confirmed/Probable bottoms:

  RSI at bottom:       Median {conf['RSI'].median():.0f}  |  Range {conf['RSI'].min():.0f} – {conf['RSI'].max():.0f}
  Drawdown from ATH:   Median {conf['Drawdown%'].median():.0f}%
  MACD at bottom:      Median hist = {conf['MACD_Hist'].median():.3f} (negative, but often just starting to contract)
  60d Forward Return:  Median {conf['Ret_60d%'].median():.0f}%  |  Avg {conf['Ret_60d%'].mean():.0f}%
  90d Forward Return:  Median {conf['Ret_90d%'].median():.0f}%  |  Avg {conf['Ret_90d%'].mean():.0f}%
""")

print("=" * 90)
print("  ANALYSIS COMPLETE")
print("=" * 90)
