from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data_loader import load_df
from lstm import PriceChangeLSTM
from macd import macd, best_macd_search
from decision_tree_regressor import decisionTreePredict, bruteForceTree, replaceNaNValue

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

# ex: http://127.0.0.1:8000/decision-tree/yahoo/MULT3.SA?start=2020-11-17&end=2021-11-17&series_size=10
# result: 
# {
#   "df_pred": 
#   {
#       "Close": ,
#       "Open": ,
#       etc,
#       .
#       .
#       .
#       "predicted": série de valores Close preditos. Os series_size primeiros valores são NaN jogados para zero,
#                    pois não tem antecessores suficientes para gerar uma série
#   },
#   "f1_score": f1 score do treinamento,
#   "next_value": o próximo valor predito de acordo com o modelo
# }
@app.get("/decision-tree/{source}/{stock}")
def get_decision_tree(source: str, stock: str, start: date, end: date, series_size: int):
    
    df = load_df(stock, source, start, end)

    if(series_size > len(df)):
        return "Not enough data for the given series size."

    # Ajusta indice e data pois as funções pressupõem que há uma coluna Dates e que o índice é numérico
    df["Dates"] = df.index
    df.index    = range(0, len(df))

    # Treina, testa e prediz
    result = decisionTreePredict(df, "Dates", "Close", series_size, 0, 0.8, False)

    # Reajusta o dataframe de resultados preditos no formato do df original
    df = df.join(result["df_pred"], lsuffix='_caller', rsuffix='_other')
    df.index = df["Dates"]
    df.drop("Dates", axis=1, inplace=True)
    result["df_pred"] = df

    # Framework não gosta de NaN, que são os valores não preditos pois não tem antecessores suficientes para gerar uma série
    result["df_pred"]["predicted"] = result["df_pred"]["predicted"].map(replaceNaNValue)
    
    return result   

# ex: http://127.0.0.1:8000/decision-tree-brutef/yahoo/MULT3.SA?start=2020-11-17&end=2021-11-17&series_size=10 #&ranges=SYNTAXE_CORRETA
# result: 
# {
#   "ranges": lista de ranges enviados,
#   "scores": lista de f1 scores (round(f1, 4)) do modelo de cada range,
#   "best_range": qual range foi o melhor,
#   "df_pred": identico ao do get_decision_tree, nesse caso é o do melhor modelo,
#   "f1_score": identico ao do get_decision_tree, nesse caso é o do melhor modelo,
#   "next_value": identico ao do get_decision_tree, nesse caso é o do melhor modelo
# }
@app.get("/decision-tree-brutef/{source}/{stock}")
def get_decision_tree_bruteforce(source: str, stock: str, start: date, end: date, ranges: list):

    df = load_df(stock, source, start, end)

    # Ajusta indice e data pois as funções pressupõem que há uma coluna Dates e que o índice é numérico
    df["Dates"] = df.index
    df.index    = range(0, len(df))

    result = bruteForceTree(df, ranges, False, False, title="")

    # Reajusta o dataframe de resultados preditos no formato do df original
    df = df.join(result["df_pred"], lsuffix='_caller', rsuffix='_other')
    df.index = df["Dates"]
    df.drop("Dates", axis=1, inplace=True)
    result["df_pred"] = df

    # Framework não gosta de NaN, que são os valores não preditos pois não tem antecessores suficientes para gerar uma série
    result["df_pred"]["predicted"] = result["df_pred"]["predicted"].map(replaceNaNValue)
    
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