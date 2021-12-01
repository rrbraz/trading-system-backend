from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data_loader import load_df
from lstm import PriceChangeLSTM
from macd import macd, best_macd_search

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/macd/{source}/{stock}")
def get_macd(source: str, stock: str, start: date, end: date, small_avg: int,
             larg_avg: int):
    df = load_df(stock, source, start, end)
    result = macd(df, window_small=small_avg, window_large=larg_avg)
    return result


@app.get("/macd-best/{source}/{stock}")
def get_macd_search(source: str, stock: str, start: date, end: date, min_window: int,
                    max_window: int, score_attr: str):
    df = load_df(stock, source, start, end)
    result = best_macd_search(df, min_window, max_window, score_attr)
    return result


@app.get("/lstm/{source}/{stock}")
def get_lstm(source: str, stock: str, start: date, end: date, epochs: int = 100,
             hidden_layer_size: int = 32, num_layers: int = 2, dropout: float = 0.2,
             lr: float = 0.01
             ):
    df = load_df(stock, source, start, end)
    print(df)
    lstm = PriceChangeLSTM(df, epochs=epochs, hidden_layer_size=hidden_layer_size,
                           num_layers=num_layers, dropout=dropout, lr=lr)
    log = lstm.train()
    evaluation = lstm.evaluate()
    next_day_prediction = lstm.predict_next_day()
    return {
        "log": log,
        "evaluation": evaluation,
        "next_day_prediction": next_day_prediction,
    }
