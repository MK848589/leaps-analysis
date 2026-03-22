# NVDA LEAPS Trading Hypothesis Document
**Created:** March 22, 2026
**Status:** Unvalidated Hypothesis — Not a Proven System
**Author:** Derived from AI-era NVDA data analysis (2023–2026)

---

> **IMPORTANT DISCLAIMER**
> This document describes a trading hypothesis built from a small dataset.
> The largest sample group contains 17 instances. The exact-match analogue
> group contains 1 instance. Nothing in this document constitutes a proven
> trading system. Every finding requires real-world validation before
> committing significant capital. Past pattern recognition is not a
> guarantee of future results.

---

## 1. THE HYPOTHESIS

We believe the following to be true based on the data:

**Core hypothesis:** NVDA, during the AI era (2023 onwards), exhibits a
measurable and exploitable pattern where fear-driven pullbacks — defined by
specific combinations of RSI compression, distance from 52-week highs,
elevated VIX, and consecutive down days — produce asymmetric recovery
returns over a 60–120 day window that make long-dated call options
(LEAPS) a positive expected-value instrument.

**Sub-hypothesis A:** The magnitude of the pullback, measured across four
indicators simultaneously, predicts the magnitude of the subsequent
recovery. Deeper fear equals larger recovery. We call this the Tier system.

**Sub-hypothesis B:** VIX level at entry is the single most important
modifier. Elevated-VIX pullbacks (VIX above 20) in the AI era produced a
60 percent bounce rate and a median 5-day return of plus 4.6 percent.
Calm-VIX pullbacks (VIX below 20) produced only a 25 percent bounce rate
and a median 5-day return of plus 0.5 percent.

**Sub-hypothesis C:** A Thursday or Friday entry, constrained by settlement
timelines, does not materially reduce expected returns because AI-era NVDA
pullbacks find their final low after Day 5 approximately 47 percent of the
time — meaning a late-week entry often captures the actual bottom rather
than missing it.

**What we believe the edge is:** Buying LEAPS during multi-indicator fear
signals captures NVDA's structural AI-era upward bias at a discount,
amplified by options leverage, with a historically favourable risk/reward
profile unavailable through stock purchase alone.

---

## 2. THE EVIDENCE

### 2a. Dataset Used
- Source: Yahoo Finance via yfinance Python library
- Universe: NVDA daily OHLCV prices, January 2023 to March 2026
- VIX data: CBOE Volatility Index, same period
- Total AI-era trading days analyzed: 806
- Indicators calculated: RSI(14), 52-week rolling high, 200-day SMA,
  20-day Bollinger Bands, ATR(14), consecutive down days

### 2b. Sample Sizes — Read This Carefully

| Finding | Sample Size | Confidence |
|---|---|---|
| Exact analogue (RSI 34–42, dist52w -13% to -22%, consec 4+, VIX 23–32) | **1 instance** | Very Low |
| Tight analogue (RSI 32–44, dist52w -12% to -25%, consec 3+, VIX 22–34) | **2 instances** | Very Low |
| Broad AI-era pullbacks (RSI 30–46, dist52w below -10%, consec 2+) | **17 instances** | Low |
| High-VIX subset (VIX above 20) | **5 instances** | Very Low |
| Tier 1 historical entries (all eras, 2015–2026) | **6 instances** | Low |
| Tier 3 historical entries (all eras) | **16 instances** | Low–Moderate |

**The honest summary:** Every number in this analysis is derived from small
samples. The 60 percent bounce rate in high-VIX conditions comes from 5
data points. The 100 percent Tier 1 hit rate comes from 6 data points.
These are observations that warrant a hypothesis — not statistics that
validate a system.

### 2c. What the Data Showed

**Tier framework results (AI era + historical, n=16–22 per tier):**
- Tier 1 entries (all 4 conditions met): median 90-day return +84.8%, 100% hit rate
- Tier 2 entries (4 conditions, slightly relaxed): median 90-day return +64.9%, 100% hit rate
- Tier 3 entries (3 of 4 conditions): median 90-day return +57.6%, 81% hit rate at 40%+
- Tier 4 entries (weak signal only): median 90-day return +24.2%, 33% hit rate at 40%+
- Baseline (random LEAPS entry): approximately +18% over 90 days

**AI-era 5-day forward return from pullback signal (n=17):**
- Bounce outcome (Day 5 above +3%): 35% of cases overall, 60% when VIX above 20
- Consolidation outcome (Day 5 between -3% and +3%): 47% of cases
- Selloff outcome (max drawdown worse than -5%): 18% of cases

**Bottom timing in AI era (n=17):**
- Bottom found within 1 day of signal: 6%
- Bottom found within 3 days: 29%
- Bottom found within 5 days: 53%
- Bottom found after Day 5: 47%

**December 2024 anchor case (the single exact analogue):**
- RSI 36.9, VIX 27.6, 5 consecutive down days, 13.4% below 52-week high
- Day 1: +1.4%, Day 3: +8.3%, Day 5: +8.5%, Day 7: +6.7%, Day 10: +12.1%
- Maximum drawdown during 10-day window: only -1.4% on Day 1
- This case resolved exactly as the hypothesis predicts — but it is one case

---

## 3. THE RULES TO TEST

These are expressed as hypotheses to be validated, not confirmed rules.

### Entry Rules (Hypothesis)

**Tier classification at entry:**

Tier 1 entry signal — all four conditions must be true simultaneously:
- RSI(14) below 35
- Price more than 20% below the rolling 52-week high
- VIX above 25
- Price more than 10% below the 200-day SMA

Tier 2 entry signal — all four conditions must be true:
- RSI(14) below 40
- Price more than 15% below the rolling 52-week high
- VIX above 22
- Price more than 5% below the 200-day SMA

Tier 3 entry signal — all three conditions must be true:
- RSI(14) below 45
- Price more than 10% below the rolling 52-week high
- Price below the 200-day SMA

No entry if none of the above tiers are met.
No entry if VIX is below 20, regardless of RSI or price levels.

### Instrument Rules (Hypothesis)
- Instrument: Long call options only (LEAPS)
- Minimum expiration: 12 months from entry date
- Target expiration: 18–24 months from entry date
- Strike selection: 5–15% out of the money at Tier 3; at the money or
  slightly in the money at Tier 1 and Tier 2
- No spreads, no short options legs

### Position Sizing Rules (Hypothesis)
- Tier 1 entry: deploy 80–100% of allocated LEAPS budget
- Tier 2 entry: deploy 60–80% of allocated LEAPS budget
- Tier 3 entry: deploy 40–50% of allocated LEAPS budget, reserve 50–60%
  for a potential Tier 1 or 2 signal on deeper weakness
- Tier 4 or no signal: 0% deployment

### Exit Rules (Hypothesis)
- Primary target: 40% gain on the option premium (based on historical
  median return profile)
- Secondary target: 80% gain on the option premium (based on upper range)
- Time-based exit: close position 60 days before expiration regardless
  of profit or loss, to avoid accelerating theta decay
- Stop-loss hypothesis: no hard stop-loss on LEAPS (option premium is
  the maximum loss; the structure itself is the stop). However, if NVDA
  falls an additional 15% from entry and no new tier signal is generated
  within 30 days, reassess the thesis

### Re-entry Rules (Hypothesis)
- After a full exit, require a minimum 30-day reset before re-entering
- A new, independent tier signal must be generated; do not re-enter on
  the same signal that triggered the original trade
- If a Tier 3 entry advances to Tier 1 conditions mid-position, this
  is an add signal — not a new position signal

---

## 4. HOW TO VALIDATE

### What Would Prove This Hypothesis Works

To have reasonable statistical confidence in this system, the following
outcomes over the next 12–24 months would support the hypothesis:

**Minimum validation threshold (weak confirmation):**
- 10 independent trades completed (entry and exit)
- Win rate above 65% (at least 7 of 10 hits the 40% gain target)
- No single loss exceeds the maximum option premium paid (which is
  structurally guaranteed by using long calls only)
- Average return across all trades exceeds 30%

**Strong validation threshold:**
- 20 independent trades completed
- Tier 1 and 2 entries achieve 80%+ win rate at the 40% gain target
- Tier 3 entries achieve 65%+ win rate at the 40% gain target
- The tier hierarchy holds: Tier 1 average return is higher than Tier 2,
  which is higher than Tier 3, which is higher than untiered entries
- VIX filter works: high-VIX entries outperform low-VIX entries

**What would disprove the hypothesis:**
- Win rate falls below 50% across 15 or more trades
- Tier 1 entries fail to outperform Tier 3 entries
- More than 3 consecutive losses at Tier 1 or 2
- A regime change in NVDA (loss of AI dominance narrative, major
  competitive displacement, regulatory restriction on AI chips) that
  invalidates the structural upward bias assumption

### How Many Trades Are Needed

Given current market conditions and NVDA's volatility, realistically
expect 4–8 Tier 3 or better entry signals per year. To reach 20 completed
trades for strong validation requires approximately 2.5 to 5 years of
live trading. This is a long validation timeline and should be factored
into capital allocation decisions.

### Tracking Metrics to Record on Every Trade
- Entry date and price
- Tier at entry (1, 2, 3, or 4)
- RSI at entry
- Distance from 52-week high at entry
- VIX at entry
- Distance from 200-day SMA at entry
- Number of consecutive down days at entry
- Option strike, expiration, and premium paid
- Exit date and price
- Reason for exit (target hit, time decay, stop, thesis broken)
- Maximum adverse excursion (how far against you before recovery)
- Maximum favorable excursion (best point reached before exit)

---

## 5. WHAT COULD BE WRONG

This section lists every known flaw and risk in the analysis.

**Statistical flaws:**

Sample sizes are too small to draw reliable conclusions. The entire
foundation of this hypothesis rests on 17 AI-era pullback instances and
1 exact analogue. In statistics, you generally need at least 30 samples
for a distribution to be meaningful and 100 or more for reliable
percentages. We have neither.

Survivorship bias is present. NVDA survived and thrived through the AI
era. We are studying a stock that we already know went up dramatically.
Had NVDA failed to maintain AI dominance, the pullbacks we studied would
have been the beginning of multi-year declines, not buying opportunities.

Overfitting risk is high. The Tier framework was built by looking at
historical data and identifying the conditions that preceded recoveries.
This is the classic data-mining problem — any set of rules built on
historical data will look better on that data than on future data.

**Regime change risks:**

The AI era may not continue. NVDA's current valuation, pricing power,
and market position depend entirely on continued AI chip demand growth.
A credible competitor, a change in AI architecture that reduces GPU
dependence, export restrictions, or a general AI spending slowdown
could fundamentally alter NVDA's mean-reversion dynamics.

The VIX relationship may not hold. The finding that high-VIX entries
outperform low-VIX entries in AI-era NVDA is based on 5 instances.
This could be coincidence rather than causation.

**Options-specific risks:**

Implied volatility expansion punishes option buyers during fear events.
When VIX is elevated and NVDA is selling off — which is precisely when
the Tier 1 signal fires — implied volatility on NVDA options spikes.
This means you pay a higher premium for the same strike and expiration
than you would in calm conditions. The analysis measured stock price
returns, not option premium returns. The actual LEAPS returns could be
significantly different from the stock return percentages used.

Time decay works against the hypothesis. LEAPS lose value every day
through theta. The analysis showed that some AI-era consolidation
scenarios chop sideways for weeks. A position entered at Tier 3
that consolidates for 4–6 weeks before resolving loses meaningful
time value even if the stock eventually recovers.

Strike selection is not validated. The hypothesis recommends specific
strike distances but there is no data comparing the performance of
different strikes across the historical signals. The optimal strike
for this strategy is unknown.

**Execution risks:**

Liquidity may be insufficient at desired strikes and expirations.
LEAPS have wider bid-ask spreads than short-dated options. The
fill price in real trading may be significantly worse than the
mid-market price used in any hypothetical return calculation.

Canadian investor currency risk is unaccounted for. The analysis
assumes USD returns. CAD to USD conversion costs and exchange rate
movement over the 60–120 day holding period can meaningfully impact
actual returns in Canadian dollar terms.

Settlement delays like the one experienced this week add execution
risk. An optimal entry signal may arise during a period when capital
is unavailable.

**Analytical assumptions that may be wrong:**

The 90-day return window is arbitrary. The analysis identified good
returns at 90 days but did not systematically test whether 60 days,
120 days, or 180 days produces better outcomes. The choice of 90 days
may be overfitted to the sample.

The four-indicator tier system is arbitrary. These four indicators
(RSI, 52-week high distance, VIX, 200-day SMA distance) were chosen
by the analyst. Other combinations of indicators might produce better
or worse results. The tier thresholds were not optimised through
systematic search but were set based on round numbers and intuition.

---

## 6. THE PAPER TRADE PLAN

### Minimum Paper Trading Period Before Full Capital Deployment

**Phase 1: Observation only (months 1 to 3)**
Do not trade. Only observe and record. Each time a Tier 3 or better
signal fires, record the entry date, price, tier, and all four
indicator values. Track what would have happened to a hypothetical
position using real-time option prices from Questrade or another
platform. The goal is to build a real-time record without the
distortions of looking at historical charts.

**Phase 2: Paper trading with real option prices (months 4 to 9)**
Execute paper trades using real bid-ask spreads from the options
market, not theoretical pricing. Record every fill as if it were
real. Include the currency conversion cost from CAD to USD. Track
against the hypothesis targets. Minimum 3 paper trades required
before proceeding to Phase 3.

**Phase 3: Reduced real capital (months 10 to 18)**
Deploy real capital at 25% of intended position sizes. This tests
execution, psychology, and actual transaction costs. The goal is
not to generate meaningful profit but to validate that the
hypothesis holds under real-money conditions.

**Phase 4: Full capital deployment (month 19 onwards)**
Only after completing Phases 1 through 3, with paper trade results
that meet the weak validation threshold (7 of 10 hits, average
return above 30%), consider deploying full intended capital.

### Exception to the Paper Trade Plan

If a Tier 1 signal fires during the observation or paper trading
phase, a small real-money position (maximum 10% of intended full
position size) is acceptable as a learning trade. Tier 1 signals
are rare enough (estimated 2–4 per year at most) that waiting for
paper trade validation may mean missing the highest-conviction
opportunities entirely. The learning value of experiencing a Tier 1
entry in real time outweighs the risk at 10% position size.

---

## 7. CONFIDENCE LEVELS

Each finding is rated 1 to 5 where:
1 = anecdotal, essentially one or two data points
2 = suggestive, small sample, directionally interesting
3 = moderate confidence, enough data to form a hypothesis worth testing
4 = reasonably confident, consistent pattern across multiple conditions
5 = high confidence, statistically robust (note: no finding reaches 5)

---

| Finding | Confidence | Reason |
|---|---|---|
| Tier 1 entries produce the highest returns | 3/5 | Consistent across eras but only 6 instances |
| Tier hierarchy (T1 > T2 > T3 > T4) is real | 3/5 | Directionally clear but small n at each tier |
| VIX above 20 improves bounce probability | 2/5 | Only 5 high-VIX AI-era instances |
| 60% bounce rate at VIX above 20 | 1/5 | 3 bounces out of 5 cases — this is not statistics |
| 47% of AI-era lows come after Day 5 | 3/5 | 17 instances, consistent finding, directionally reliable |
| December 2024 as the anchor analogue | 2/5 | One case — informative but not predictive |
| 200-day SMA as Tier boundary | 3/5 | Well-established technical level, consistent signal |
| RSI 35 as Tier 1 threshold | 2/5 | Threshold is intuitive, not data-optimised |
| 90-day return window is optimal | 2/5 | Tested but not systematically compared to alternatives |
| Thursday entry is not disadvantaged | 3/5 | Supported by bottom-timing data across 17 cases |
| $177.50 as the Thursday decision level | 4/5 | Derived from hard Tier boundaries, not curve-fitted |
| Tier 1 has 100% hit rate | 2/5 | 6 instances only — one bad outcome changes this to 83% |
| AI-era NVDA outperforms mean after signals | 3/5 | Consistent direction but magnitude uncertain |
| Options returns mirror stock returns | 1/5 | Not tested at all — critical unknown |

### Most Reliable Findings (use these with reasonable confidence)
- The Tier framework correctly identifies fear vs. non-fear entries.
  Even if the exact return numbers are overstated, the directional
  ranking is likely to hold: deeper fear equals better entry.
- The $177.50 Thursday decision threshold is mechanically sound.
  It is derived from fixed indicator boundaries, not historical
  curve-fitting.
- Bottom timing in the AI era is slower than pre-AI era. The 47%
  after-Day-5 finding is robust across 17 instances and consistent
  with NVDA's increased institutional attention slowing capitulation.

### Least Reliable Findings (treat as directional signals only)
- Any specific return percentage. The medians and means are based
  on tiny samples. The actual distribution of outcomes is unknown.
- The VIX regime split. Based on 5 instances in the high-VIX group.
  Directionally plausible but not statistically meaningful.
- Option-level returns. The entire analysis was done on stock prices.
  The translation to option premium performance has not been validated
  and is subject to implied volatility dynamics that could materially
  alter results in either direction.

---

## FINAL STATEMENT ON THIS DOCUMENT

This hypothesis was built carefully and honestly from the available
data. The methodology is sound for what it is: exploratory analysis
that generates a testable hypothesis. It is not a backtest of a
confirmed system. It is not financial advice. It should not be used
to justify deploying more capital than you are prepared to lose
entirely, because options can go to zero.

The most useful function of this document is to create a structured
framework that forces discipline: defined entry criteria, defined
position sizing, defined exits, and a defined validation process.
That structure is valuable regardless of whether the underlying
hypothesis ultimately proves correct.

Review and update this document after every 5 completed trades.
If the results consistently contradict the hypothesis, abandon it.

---

*Document version: 1.0*
*Next review: After 5 completed trades or September 2026, whichever comes first*
