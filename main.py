from typing import Optional

from fastapi import FastAPI
from datetime import date

from macd import load_df, macd, best_macd_search

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/macd/{source}/{stock}")
def get_macd(source: str, stock: str, start: date, end: date, small_avg: int, larg_avg: int):
    df = load_df(stock, source, start, end)
    result = macd(df, window_small=small_avg, window_large=larg_avg)
    return result


@app.get("/macd-best/{source}/{stock}")
def get_macd(source: str, stock: str, start: date, end: date, min_window: int, max_window: int, score_attr: str):
    df = load_df(stock, source, start, end)
    result = best_macd_search(df, min_window, max_window, score_attr)
    return result
