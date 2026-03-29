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

        # ── Measure price action after signal ────────────────────────────────
        post_signal = day_df[
            (day_df.index.time > SIGNAL_TIME) &
            (day_df.index.time <= TRADE_END_TIME)
        ]
        if post_signal.empty:
            continue

        end_price  = float(post_signal['Close'].iloc[-1])
        high_price = float(post_signal['High'].max())
        low_price  = float(post_signal['Low'].min())

        price_change     = end_price - signal_price
        price_change_pct = (price_change / signal_price) * 100
        max_move_up      = ((high_price - signal_price) / signal_price) * 100
        max_move_down    = ((low_price  - signal_price) / signal_price) * 100

        # ── Did trend continue in expected direction? ─────────────────────────
        if stack == 'bearish':
            trend_continued = price_change < 0
            max_favorable   = max_move_down
        elif stack == 'bullish':
            trend_continued = price_change > 0
            max_favorable   = max_move_up
        else:
            trend_continued = None
            max_favorable   = None

        results.append({
            'date'             : date,
            'stack'            : stack,
            'signal_price'     : round(signal_price, 2),
            'end_price'        : round(end_price, 2),
            'price_change'     : round(price_change, 2),
            'price_change_pct' : round(price_change_pct, 2),
            'max_move_up_pct'  : round(max_move_up, 2),
            'max_move_down_pct': round(max_move_down, 2),
            'max_favorable_pct': round(max_favorable, 2) if max_favorable is not None else None,
            'trend_continued'  : trend_continued,
            'ema9'             : round(to_scalar(signal_bar['EMA9']),   2),
            'ema48'            : round(to_scalar(signal_bar['EMA48']),  2),
            'ema200'           : round(to_scalar(signal_bar['EMA200']), 2),
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
            winners    = subset[subset['trend_continued'] == True]
            win_rate   = len(winners) / len(subset) * 100
            avg_move   = subset['price_change_pct'].mean()
            avg_fav    = subset['max_favorable_pct'].mean()
            avg_unfav  = subset['max_move_up_pct'].mean() if stack_type == 'bearish' else subset['max_move_down_pct'].mean()

            print(f"  Win rate (trend continued):  {win_rate:.1f}%")
            print(f"  Avg price change by 3:30pm:  {avg_move:.2f}%")
            print(f"  Avg max favorable move:      {avg_fav:.2f}%")
            print(f"  Avg max adverse move:        {avg_unfav:.2f}%")
            print(f"  Best day:                    {subset['price_change_pct'].min() if stack_type == 'bearish' else subset['price_change_pct'].max():.2f}%")
            print(f"  Worst day:                   {subset['price_change_pct'].max() if stack_type == 'bearish' else subset['price_change_pct'].min():.2f}%")
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

if __name__ == "__main__":
    main()
