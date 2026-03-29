import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER         = "NVDA"
INTERVAL       = "5m"
PERIOD         = "60d"          # yfinance free tier max for 5m
SIGNAL_TIME    = time(10, 30)   # EMA stack check time
TRADE_END_TIME = time(15, 30)   # measure move until here
EMA_SHORT      = 9
EMA_MID        = 48
EMA_LONG       = 200

# ─────────────────────────────────────────────────────────────────────────────

def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]   # drop duplicate cols from MultiIndex flatten
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
    return df

def add_emas(df):
    df = df.copy()
    df['EMA9']   = df['Close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['EMA48']  = df['Close'].ewm(span=EMA_MID,   adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=EMA_LONG,  adjust=False).mean()
    return df

def to_scalar(val):
    """Safely extract a Python float from any pandas/numpy scalar, Series, or array."""
    return float(np.array(val).flat[0])

def classify_stack(row):
    """Returns 'bearish', 'bullish', or 'neutral' based on EMA order."""
    e9   = to_scalar(row['EMA9'])
    e48  = to_scalar(row['EMA48'])
    e200 = to_scalar(row['EMA200'])
    if e9 < e48 < e200:
        return 'bearish'
    elif e9 > e48 > e200:
        return 'bullish'
    else:
        return 'neutral'

def run_backtest(df):
    results = []
    trading_days = df.groupby(df.index.date)

    for date, day_df in trading_days:
        # ── Get signal bar at 10:30 ──────────────────────────────────────────
        signal_bars = day_df[
            (day_df.index.time >= SIGNAL_TIME) &
            (day_df.index.time < time(10, 35))
        ]
        if signal_bars.empty:
            continue

        signal_bar   = signal_bars.iloc[0]
        stack        = classify_stack(signal_bar)
        signal_price = to_scalar(signal_bar['Close'])
        signal_open  = to_scalar(signal_bar['Open'])
        signal_low   = to_scalar(signal_bar['Low'])

        # ── Continuation filters ─────────────────────────────────────────────

        # 1. Drop from open: how much has price already moved by 10:30?
        open_bar      = day_df[day_df.index.time < time(9, 35)]
        day_open      = to_scalar(open_bar['Open'].iloc[0]) if not open_bar.empty else signal_price
        open_to_signal_pct = ((signal_price - day_open) / day_open) * 100

        # 2. First-hour direction: was price still falling into the signal bar?
        #    Compare 10:00 AM close to 10:30 AM close
        bar_1000 = day_df[
            (day_df.index.time >= time(10, 0)) &
            (day_df.index.time < time(10, 5))
        ]
        price_1000 = to_scalar(bar_1000['Close'].iloc[0]) if not bar_1000.empty else signal_price
        still_falling = signal_price < price_1000   # bearish: price lower at 10:30 than 10:00
        still_rising  = signal_price > price_1000   # bullish: price higher at 10:30 than 10:00

        # 3. New intraday low at signal: 10:30 bar low < all lows since open
        pre_signal = day_df[day_df.index.time < SIGNAL_TIME]
        prior_low  = to_scalar(pre_signal['Low'].min()) if not pre_signal.empty else signal_low
        prior_high = to_scalar(pre_signal['High'].max()) if not pre_signal.empty else signal_open
        is_new_low  = signal_low  <= prior_low   # bearish continuation
        is_new_high = to_scalar(signal_bar['High']) >= prior_high  # bullish continuation

        # 4. Signal bar direction: is the 10:30 bar itself red (bearish) or green?
        signal_bar_red   = signal_price < signal_open
        signal_bar_green = signal_price > signal_open

        # ── Measure price action after signal ────────────────────────────────
        post_signal = day_df[
            (day_df.index.time > SIGNAL_TIME) &
            (day_df.index.time <= TRADE_END_TIME)
        ]
        if post_signal.empty:
            continue

        end_price  = to_scalar(post_signal['Close'].iloc[-1])
        high_price = to_scalar(post_signal['High'].max())
        low_price  = to_scalar(post_signal['Low'].min())

        price_change     = end_price - signal_price
        price_change_pct = (price_change / signal_price) * 100
        max_move_up      = ((high_price - signal_price) / signal_price) * 100
        max_move_down    = ((low_price  - signal_price) / signal_price) * 100

        # ── Did trend continue in expected direction? ─────────────────────────
        if stack == 'bearish':
            trend_continued   = price_change < 0
            max_favorable     = max_move_down
            # Continuation filter: still falling + new low + red bar + not exhausted (drop < 2%)
            continuation_pass = (still_falling and is_new_low and signal_bar_red
                                  and open_to_signal_pct > -2.0)
        elif stack == 'bullish':
            trend_continued   = price_change > 0
            max_favorable     = max_move_up
            continuation_pass = (still_rising and is_new_high and signal_bar_green
                                  and open_to_signal_pct < 2.0)
        else:
            trend_continued   = None
            max_favorable     = None
            continuation_pass = False

        results.append({
            'date'                : date,
            'stack'               : stack,
            'signal_price'        : round(signal_price, 2),
            'end_price'           : round(end_price, 2),
            'price_change'        : round(price_change, 2),
            'price_change_pct'    : round(price_change_pct, 2),
            'max_move_up_pct'     : round(max_move_up, 2),
            'max_move_down_pct'   : round(max_move_down, 2),
            'max_favorable_pct'   : round(max_favorable, 2) if max_favorable is not None else None,
            'trend_continued'     : trend_continued,
            'continuation_pass'   : continuation_pass,
            'open_to_signal_pct'  : round(open_to_signal_pct, 2),
            'still_falling'       : still_falling,
            'is_new_low'          : is_new_low,
            'signal_bar_red'      : signal_bar_red,
            'ema9'                : round(to_scalar(signal_bar['EMA9']),   2),
            'ema48'               : round(to_scalar(signal_bar['EMA48']),  2),
            'ema200'              : round(to_scalar(signal_bar['EMA200']), 2),
        })

    return pd.DataFrame(results)

def print_summary(results_df):
    print("\n" + "="*60)
    print(f"NVDA 5-MIN EMA STACK BACKTEST — {TICKER}")
    print(f"Signal time: {SIGNAL_TIME} | Period: {PERIOD}")
    print("="*60)

    total_days = len(results_df)
    print(f"\nTotal trading days analyzed: {total_days}")

    for stack_type in ['bearish', 'bullish', 'neutral']:
        subset = results_df[results_df['stack'] == stack_type]
        if subset.empty:
            continue

        print(f"\n── {stack_type.upper()} STACK ({len(subset)} days) ──────────────")

        if stack_type != 'neutral':
            def stack_stats(s, label):
                if s.empty:
                    print(f"  [{label}] No days match.")
                    return
                fav     = s['max_favorable_pct'].abs()
                adv     = s['max_move_up_pct'].abs() if stack_type == 'bearish' else s['max_move_down_pct'].abs()
                wr_eod  = (s['trend_continued'] == True).sum() / len(s) * 100
                wr_50   = (fav >= 0.50).sum() / len(s) * 100
                wr_75   = (fav >= 0.75).sum() / len(s) * 100
                wr_100  = (fav >= 1.00).sum() / len(s) * 100
                wr_150  = (fav >= 1.50).sum() / len(s) * 100
                wr_rr   = (fav > adv).sum()   / len(s) * 100
                print(f"  [{label}]  n={len(s)}")
                print(f"    Win rate (EOD direction):  {wr_eod:.0f}%")
                print(f"    Win rate (fav ≥ 0.50%):    {wr_50:.0f}%")
                print(f"    Win rate (fav ≥ 0.75%):    {wr_75:.0f}%")
                print(f"    Win rate (fav ≥ 1.00%):    {wr_100:.0f}%")
                print(f"    Win rate (fav ≥ 1.50%):    {wr_150:.0f}%")
                print(f"    Win rate (fav > adverse):  {wr_rr:.0f}%")
                print(f"    Avg max favorable:         {s['max_favorable_pct'].mean():.2f}%")
                print(f"    Avg EOD move:              {s['price_change_pct'].mean():.2f}%")

            stack_stats(subset, "ALL days")
            filtered = subset[subset['continuation_pass'] == True]
            print(f"  ── Continuation filters: still falling + new low + red bar + drop<2% ──")
            stack_stats(filtered, "FILTERED days")

            # Per-filter breakdown to show which filters help most
            if stack_type == 'bearish' and len(subset) > 0:
                n = len(subset)
                print(f"  ── Individual filter pass rates ──")
                print(f"    Still falling (10:00→10:30): {subset['still_falling'].sum()}/{n} ({subset['still_falling'].mean()*100:.0f}%)")
                print(f"    New intraday low at 10:30:   {subset['is_new_low'].sum()}/{n} ({subset['is_new_low'].mean()*100:.0f}%)")
                print(f"    Signal bar is red:           {subset['signal_bar_red'].sum()}/{n} ({subset['signal_bar_red'].mean()*100:.0f}%)")
                not_exhausted = (subset['open_to_signal_pct'] > -2.0).sum()
                print(f"    Drop from open < 2%:         {not_exhausted}/{n} ({not_exhausted/n*100:.0f}%)")
        else:
            print(f"  Avg price change by 3:30pm:  {subset['price_change_pct'].mean():.2f}%")

    # ── Valid trading days per month ─────────────────────────────────────────
    directional = results_df[results_df['stack'] != 'neutral']
    print(f"\n── TRADING OPPORTUNITY FREQUENCY ──────────────────────")
    print(f"  Directional stack days (non-neutral): {len(directional)} / {total_days} ({len(directional)/total_days*100:.1f}%)")

    directional = directional.copy()
    directional['month'] = pd.to_datetime(directional['date']).dt.to_period('M')
    monthly = directional.groupby('month').size()
    print(f"  Avg directional days per month:       {monthly.mean():.1f}")
    print(f"  Min in a month:                       {monthly.min()}")
    print(f"  Max in a month:                       {monthly.max()}")
    print("\n" + "="*60)

def find_ema_rejections(df):
    """
    BEARISH stack days: price bounces into EMA9/48 from below, closes under → short entry.
    BULLISH stack days: price dips into EMA9/48 from above, closes back above → long entry.
    Scan post-10:30 bars for first occurrence each day.
    """
    rejections = []
    trading_days = df.groupby(df.index.date)

    for date, day_df in trading_days:
        signal_bars = day_df[
            (day_df.index.time >= SIGNAL_TIME) &
            (day_df.index.time < time(10, 35))
        ]
        if signal_bars.empty:
            continue

        stack = classify_stack(signal_bars.iloc[0])
        if stack not in ('bearish', 'bullish'):
            continue

        post = day_df[
            (day_df.index.time > SIGNAL_TIME) &
            (day_df.index.time <= time(14, 30))
        ]

        rejection_bar = None
        rejection_ema = None

        for bar_time, bar in post.iterrows():
            hi    = to_scalar(bar['High'])
            lo    = to_scalar(bar['Low'])
            close = to_scalar(bar['Close'])
            e9    = to_scalar(bar['EMA9'])
            e48   = to_scalar(bar['EMA48'])

            if stack == 'bearish':
                # Price touched EMA from below but closed under it → resistance confirmed
                if hi >= e48 and close < e48:
                    rejection_bar, rejection_ema = bar, 'EMA48'
                    break
                elif hi >= e9 and close < e9:
                    rejection_bar, rejection_ema = bar, 'EMA9'
                    break
            else:  # bullish
                # Price dipped into EMA from above but closed back above → support confirmed
                if lo <= e48 and close > e48:
                    rejection_bar, rejection_ema = bar, 'EMA48'
                    break
                elif lo <= e9 and close > e9:
                    rejection_bar, rejection_ema = bar, 'EMA9'
                    break

        if rejection_bar is None:
            rejections.append({
                'date': date, 'stack': stack, 'rejection_found': False,
                'rejection_ema': None, 'entry_time': None, 'entry_price': None,
                'max_favorable_pct': None, 'eod_pct': None,
                'trend_continued': None, 'hit_050': None, 'hit_075': None, 'hit_100': None,
            })
            continue

        entry_price = to_scalar(rejection_bar['Close'])
        entry_time  = bar_time.time()

        after_entry = day_df[
            (day_df.index.time > entry_time) &
            (day_df.index.time <= TRADE_END_TIME)
        ]
        if after_entry.empty:
            continue

        eod_price = to_scalar(after_entry['Close'].iloc[-1])
        eod_pct   = ((eod_price - entry_price) / entry_price) * 100

        if stack == 'bearish':
            low_after = to_scalar(after_entry['Low'].min())
            max_fav   = ((low_after   - entry_price) / entry_price) * 100
            win       = eod_pct < 0
            hit_050   = max_fav <= -0.50
            hit_075   = max_fav <= -0.75
            hit_100   = max_fav <= -1.00
        else:  # bullish
            high_after = to_scalar(after_entry['High'].max())
            max_fav    = ((high_after  - entry_price) / entry_price) * 100
            win        = eod_pct > 0
            hit_050    = max_fav >= 0.50
            hit_075    = max_fav >= 0.75
            hit_100    = max_fav >= 1.00

        rejections.append({
            'date'             : date,
            'stack'            : stack,
            'rejection_found'  : True,
            'rejection_ema'    : rejection_ema,
            'entry_time'       : entry_time,
            'entry_price'      : round(entry_price, 2),
            'max_favorable_pct': round(max_fav, 2),
            'eod_pct'          : round(eod_pct, 2),
            'trend_continued'  : win,
            'hit_050'          : hit_050,
            'hit_075'          : hit_075,
            'hit_100'          : hit_100,
        })

    return pd.DataFrame(rejections)


def print_rejection_summary(rej_df):
    total_days  = len(rej_df)
    found       = rej_df[rej_df['rejection_found'] == True]
    total_found = len(found)

    print("\n" + "="*60)
    print("EMA REJECTION ENTRY — BEARISH + BULLISH STACK DAYS")
    print("  Bearish: price touches EMA9/48 from below, closes under → short")
    print("  Bullish: price dips into EMA9/48 from above, closes back over → long")
    print("="*60)

    # ── Overall frequency ───────────────────────────────────────────────────
    bear_all   = rej_df[rej_df['stack'] == 'bearish']
    bull_all   = rej_df[rej_df['stack'] == 'bullish']
    bear_found = found[found['stack'] == 'bearish']
    bull_found = found[found['stack'] == 'bullish']

    print(f"\n  ── FREQUENCY (over {total_days} directional stack days) ──")
    print(f"  Bearish stack days:          {len(bear_all)}  → rejections found: {len(bear_found)} ({len(bear_found)/max(len(bear_all),1)*100:.0f}%)")
    print(f"  Bullish stack days:          {len(bull_all)}  → rejections found: {len(bull_found)} ({len(bull_found)/max(len(bull_all),1)*100:.0f}%)")
    print(f"  Total rejection signals:     {total_found} over 60 days  (~{total_found/3:.1f}/week | ~{total_found*4/3:.0f}/month)")

    # EMA9 vs EMA48 split across all
    print(f"\n  EMA9  rejections (all):      {(found['rejection_ema']=='EMA9').sum()}")
    print(f"  EMA48 rejections (all):      {(found['rejection_ema']=='EMA48').sum()}")

    times = pd.to_datetime(found['entry_time'].astype(str))
    print(f"  Avg entry time:              {times.mean().strftime('%H:%M')}")
    print(f"  Entry time range:            {times.min().strftime('%H:%M')} – {times.max().strftime('%H:%M')}")

    # ── Per-stack win rates ──────────────────────────────────────────────────
    for stack_type, subset, direction in [
        ('BEARISH (short)', bear_found, 'down'),
        ('BULLISH (long)',  bull_found, 'up'),
    ]:
        if subset.empty:
            continue
        n = len(subset)
        e9s  = subset[subset['rejection_ema'] == 'EMA9']
        e48s = subset[subset['rejection_ema'] == 'EMA48']

        wr_eod = (subset['trend_continued'] == True).sum() / n * 100
        wr_050 = subset['hit_050'].sum() / n * 100
        wr_075 = subset['hit_075'].sum() / n * 100
        wr_100 = subset['hit_100'].sum() / n * 100

        print(f"\n  ── {stack_type}  (n={n}) ──")
        print(f"    EOD direction win rate:    {wr_eod:.0f}%")
        print(f"    Hit 0.50% target:          {wr_050:.0f}%")
        print(f"    Hit 0.75% target:          {wr_075:.0f}%")
        print(f"    Hit 1.00% target:          {wr_100:.0f}%")
        print(f"    Avg max favorable:         {subset['max_favorable_pct'].mean():.2f}%")
        print(f"    Avg EOD from entry:        {subset['eod_pct'].mean():.2f}%")

        for ema_label, es in [('EMA9', e9s), ('EMA48', e48s)]:
            if len(es) < 2:
                continue
            wr = es['hit_050'].sum() / len(es) * 100
            print(f"    [{ema_label}] n={len(es)}  Hit 0.50%: {wr:.0f}%  Avg fav: {es['max_favorable_pct'].mean():.2f}%  Avg EOD: {es['eod_pct'].mean():.2f}%")

    # ── Recent signals ───────────────────────────────────────────────────────
    print(f"\n  ── Recent signals (last 10) ──")
    cols = ['date','stack','rejection_ema','entry_time','entry_price','max_favorable_pct','eod_pct','trend_continued']
    print(found[cols].tail(10).to_string(index=False))
    print("="*60)

    out_path = "nvda_ema_rejection_results.csv"
    found.to_csv(out_path, index=False)
    print(f"\nRejection results saved to: {out_path}")


def main():
    print(f"Fetching {TICKER} {INTERVAL} data...")
    df = get_data(TICKER, PERIOD, INTERVAL)
    df = add_emas(df)

    print(f"Running backtest...")
    results = run_backtest(df)

    if results.empty:
        print("No results generated. Check data.")
        return

    print_summary(results)

    out_path = "nvda_ema_stack_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")
    print("\nSample rows (last 10):")
    print(results[['date','stack','signal_price','price_change_pct','max_favorable_pct','trend_continued']].tail(10).to_string(index=False))

    # ── EMA Rejection entry analysis ────────────────────────────────────────
    print(f"\nScanning for EMA rejection entries on bearish stack days...")
    rej_results = find_ema_rejections(df)
    print_rejection_summary(rej_results)

if __name__ == "__main__":
    main()
