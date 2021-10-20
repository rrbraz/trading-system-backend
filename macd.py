import datetime

import numpy as np
import requests_cache
from pandas_datareader import data as web


def load_df(stock, source, start, end):
    expire_after = datetime.timedelta(hours=1)
    session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)
    session.headers = {'User-Agent': 'insomnia/2021.5.2', 'Accept': 'application/json;charset=utf-8'}

    df = web.DataReader(stock, source, start=start, end=end, session=session)
    return df


def avaliate_macd(df, window_small, window_large):
    avg_small = df.Close.ewm(span=window_small, min_periods=window_small).mean()

    # Calculate and define moving average of window_large periods
    avg_large = df.Close.ewm(span=window_large, min_periods=window_large).mean()
    df['Trading Sign'] = np.where(avg_small > avg_large, 1, -1)
    df['Transaction'] = df['Trading Sign'].rolling(2).apply(lambda v: -v.prod(), raw=True)
    df['Daily Change'] = df.Close - df.Close.shift(1)
    df['Daily Trading Profit'] = df['Trading Sign'] * df['Daily Change']

    tdf = df[df.Transaction == 1].copy()
    tdf['Position'] = tdf.Close * tdf['Trading Sign']
    tdf['Profit/Loss'] = (tdf.Position * -1).rolling(2).sum()
    tdf['Winning Trades'] = np.where(tdf['Profit/Loss'] > 0, 1, 0)
    cumulative_return = tdf['Profit/Loss'].sum() / tdf.Close.iloc[0]
    winning_trades_rate = np.average(tdf['Profit/Loss'] > 0)
    n_transactions = np.sum(df.Transaction > 0)
    up_periods = np.sum(df['Daily Trading Profit'] > 0)
    down_periods = np.sum(df['Daily Trading Profit'] < 0)

    return {
        "cumulative_return": cumulative_return.item(),
        "winning_trades_rate": winning_trades_rate.item(),
        "n_transactions": n_transactions.item(),
        "up_periods": up_periods.item(),
        "down_periods": down_periods.item()
    }


def macd(df, window_small=12, window_large=26):
    # Calculate and define moving average of window_small periods
    avg_small = df.Close.ewm(span=window_small, min_periods=0).mean()

    # Calculate and define moving average of window_large periods
    avg_large = df.Close.ewm(span=window_large, min_periods=0).mean()

    trace_small = {
        'x': list(df.index),
        'y': list(avg_small),
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'blue'
        },
        'name': f'Moving Average of {window_small} periods'
    }

    trace_large = {
        'x': list(df.index),
        'y': list(avg_large),
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'red'
        },
        'name': f'Moving Average of {window_large} periods'
    }

    macd = avg_small - avg_large

    trace_macd = {
        'name': 'MACD',
        'x': list(df.index),
        'y': macd,
    }

    avaliation = avaliate_macd(df, window_small, window_large)

    return {
        'trace_small': trace_small,
        'trace_large': trace_large,
        'trace_macd': trace_macd,
        'avaliation': avaliation
    }


def best_macd_search(df, min_window, max_window, score_attr):
    highest_score = float('-inf')
    best_pair = None

    for window_small in range(min_window, max_window-1):
        for window_large in range(window_small + 1, max_window):
            avaliation = avaliate_macd(df, window_small, window_large)
            score = avaliation[score_attr]

            if score > highest_score:
                highest_score = score
                best_pair = (window_small, window_large)

    avaliation = avaliate_macd(df, best_pair[0], best_pair[1])
    return {
        'window_small': best_pair[0],
        'window_large': best_pair[1],
        'avaliation': avaliation
    }
