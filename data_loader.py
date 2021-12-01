import datetime
import os
from io import StringIO

import pandas as pd
import requests
import requests_cache
from pandas_datareader import data as web


def load_df(stock, source, start, end):
    expire_after = datetime.timedelta(hours=1)
    session = requests_cache.CachedSession(cache_name='cache', backend='sqlite',
                                           expire_after=expire_after)
    session.headers = {
        'User-Agent': 'insomnia/2021.5.2',
        'Accept': 'application/json;charset=utf-8'
    }

    if source.startswith("av-"):
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    else:
        api_key = None

    if source == "av-intraday":
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock}&interval=15min&apikey={api_key}&outputsize=full&datatype=csv'
        r = requests.get(url)
        df = pd.read_csv(StringIO(r.text), index_col=0)
    else:
        df = web.DataReader(stock, source, start=start, end=end, session=session)

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)

    return df
