#import matplotlib.pyplot    as plt

from   timeit import default_timer as timer
import pandas as pd
import numpy  as np
import math

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree            import DecisionTreeRegressor
from sklearn.metrics         import f1_score

# # # # # # # # # # # # # # # # Transformações # # # # # # # # # # # # # # # #
def replaceNaNValue(x, val=0):
    
    if math.isnan(x):
        return val
    else:
        return x

def replaceNaN(df, cols):
    
    for col in cols:
        df[col] = df[col].map(replaceNaNValue)
        
    return df

def toHighLow(x):
    return "high" if x >= 0 else "low"

def toHighLowBin(x):
    return 1 if x >= 0 else 0

def toHighLowDf(df, columns):

    hl = lambda x: "high" if x >= 0 else "low"
    
    for col in columns:
        series = pd.DataFrame(np.nan, index=df.index, columns=[col + "_hl"])
        for i in range(1, len(df)):
            series[col + "_hl"][i] = hl(df[col][i] - df[col][i-1])
            
        df = pd.concat([df, series], axis=1)
        
    return df.iloc[1:, :]

def ilr(x, x_prev):
    return math.log(x + 0.0001) - math.log(x_prev + 0.0001)

def invilr(x_irl, x_prev):
    return math.e**(x_irl) * x_prev

def toILR(df, columns):
    
    for col in columns:
        series = pd.DataFrame(np.nan, index=df.index, columns=[col + "_ilr"])
        for i in range(1, len(df)):
            series[col + "_ilr"][i] = ilr(df[col][i], df[col][i-1])
            if(abs(series[col + "_ilr"][i]) > 1):
                series[col + "_ilr"][i] = 1
        
        df = pd.concat([df, series], axis=1)
        
    return {"ilr": df.iloc[1:, :], "name": "_ilr"}

def rebuildFromILR(df_ilr,  df_original, regression_column, rebuilt_column, gap = 0):
    
    rebuilt = pd.DataFrame(np.nan, df_ilr.index, columns=["predicted"])
    
    for i in range(min(df_ilr.index), max(df_ilr.index)+1):
        rebuilt["predicted"][i] = invilr(df_ilr[regression_column].loc[i], df_original[rebuilt_column][i-1-gap])
        
    return rebuilt

# Gera shift_rows novas colunas para cada coluna, shiftadas sequencialmente entre 1 e shift_rows linhas
def seriesToColumns(df, columns, shift_rows, gap=0):

    # Cria shift_row novas colunas, cada uma deslocada entre 1 e shift_row além do gap
    for col in columns:
        for i in range(1, shift_rows):
            series       = df[col].shift(gap + i)
            series.name  = col + str(i)
            df = pd.concat([df, series], axis=1)
        
    # Remove qualquer linha que contenha novos valores NaN
    return df.iloc[(shift_rows + gap):, :]

# # # # # # # # # # # # # # # # Arvore de Decisao # # # # # # # # # # # # # # # #

def predNext(model, df, original_cols, series_size, gap):
    series_columns = df.columns.drop(original_cols)
    pred_df        = df.iloc[-series_size-gap:, :]
    series         = seriesToColumns(pred_df[pred_df.columns.drop(original_cols)], series_columns, series_size, gap-1)
    series         = series[series.columns.drop(series_columns)].iloc[-1:, :]
    return model.predict(series)

def testModel(res, regression_column, test_range, flag_show=True):
    
    y_true = res["df_true"][regression_column].loc[test_range:].map(toHighLow)
    y_pred = res["df_pred"][regression_column].loc[test_range:].map(toHighLow)
    score  = f1_score(y_true, y_pred, average='weighted')
        
    return score

def treeSeriesRegression(data_frame, regression_column, series_size, time_gap, test_range):
    
    df = data_frame.copy()
    columns        = df.columns
    series_columns = columns
    
    # Gera colunas de series de dados
    df = seriesToColumns(df, series_columns, series_size, time_gap)
    df.drop(columns.drop(regression_column), axis=1, inplace=True)

    # Separa linhas de treino e de teste
    dfTrain = df.iloc[:(test_range-series_size - time_gap), :]
    dfTest  = df.iloc[(test_range-series_size - time_gap):, :] 

    # Separa coluna de regressão para o treino
    xTrain = dfTrain.drop([regression_column], axis=1)
    yTrain = dfTrain[regression_column]

    # Separa coluna de regressão para o teste
    xTest = dfTest.drop([regression_column], axis=1)
    yTest = dfTest[regression_column]

    # Instancia arvore de decisão e treina
    dtr = DecisionTreeRegressor(splitter="best", random_state=138274724)
    dtr.fit(xTrain, yTrain)
    
    # Usa o modelo para prever
    yPredTrain = dtr.predict(xTrain)
    yPredTest  = dtr.predict(xTest)

    yPredTrain = pd.DataFrame(yPredTrain)
    yPredTest  = pd.DataFrame(yPredTest)

    yPred = yPredTrain.append(yPredTest, ignore_index=True)
    yPred.sort_index(inplace=True)
    yPred.columns = [regression_column]
    
    yPred.index = yPred.index.map(lambda x: x + series_size + time_gap + 1)
    
    return {"model": dtr, "df_true": df, "df_pred": yPred}

def decisionTreePredict(data_frame, date_column, target_column, series_size, gap, test_ratio, flag_print=True):
    df   = data_frame.copy()
    cols = df.columns

    # Remove NaN indesejados
    df   = replaceNaN(df, df.columns.drop(date_column))

    # Calcula ILR das colunas numericas
    res  = toILR(df, df.columns.drop(date_column))
    regression_column = target_column + res["name"]
    df   = res["ilr"]

    # Calcula indice que divide os dados de treino e de teste
    test_range  = int(len(df)*test_ratio)

    # Treina e testa o modelo
    res  = treeSeriesRegression(df[df.columns.drop(cols)], regression_column, series_size, gap, test_range)

    # Gera o f1 score do teste
    test_score = testModel(res, regression_column, test_range)

    # Reconstroi os valores do alvo se baseando nos ilrs preditos
    rebuilt = rebuildFromILR(res["df_pred"], data_frame, regression_column, target_column)

    # Usa o modelo para prever o proximo ilr da serie
    nextILR = predNext(res["model"], df, cols, series_size, gap)

    # Usa o proximo ilr predito para gerar o valor da variavel alvo predito
    nextVal = invilr(nextILR, data_frame[target_column].iloc[-1:])

    # Gera um plot do 
    if flag_print:
        rebuilt["predicted"].loc[test_range:].plot()
        data_frame[target_column].loc[test_range:].plot()
        plt.show()

    #return {"model": res["model"], "df_true_ilr": res["df_true"], "df_pred_ilr": res["df_pred"], "df_pred": rebuilt, "next_value": nextVal, "f1_score": test_score}
    return {"df_pred": rebuilt, "next_value": nextVal, "f1_score": test_score}

def bruteForceTree(data_frame, ranges, plotv, text, title=""):
    
    digits    = int(math.log10(max(ranges)))+1
    times     = []
    f1s       = []
    resarr    = []
    bestindex = 0

    for i in range(0, len(ranges)):
        start = timer()
        res   = decisionTreePredict(data_frame, "Dates", "Close", ranges[i], 0, 0.8, False)
        resarr.append(res)
        dt = timer() - start
        f1 = round(res["f1_score"], 4)
        times.append(dt)
        f1s.append(f1)

        if(f1 > f1s[bestindex]):
            bestindex = i

        if text:
            print("range: " + str(ranges[i]).zfill(digits) + ", f1: %.3f " %f1  + ", t: %.3fs" % dt)

    if text:
        print("mean time: %.3fs" % np.mean(times))
        print("mean f1  : %.3f" % np.mean(f1s))
        print("best f1  : %.3f" % f1s[bestindex] + ", range: " + str(ranges[bestindex]).zfill(digits))

    if plotv:
        plt.plot(ranges, f1s)
        plt.title(title)
        plt.show()
        
    return {"ranges": ranges, "scores": f1s, "best_range": ranges[bestindex],
            "df_pred": resarr[bestindex]["df_pred"], "f1_score": resarr[bestindex]["f1_score"],
            "next_value": resarr[bestindex]["next_value"]}