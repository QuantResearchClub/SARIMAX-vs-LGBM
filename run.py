from Functions import *
from Prediction import *


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from datetime import date
from sklearn.model_selection import TimeSeriesSplit


import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------

n_fold = 3 
folds = TimeSeriesSplit(
    n_splits = n_fold,
    gap=0, # gap between the train and test side of the splits. 
    max_train_size=10000, 
    test_size=10, # about 10% of training data
)



"""
- desti: A more detailed sepration name of dataset if needed.
    for example, curr_ts = str(time.time()).split(".")[0] is a kind of desti.
"""
desti = ""
myHELP = HelpFunctions()
myTrainer = TimeSeriesTraining()
# myHELP.yf_downloadData(VS_TS, EF_TS, 3, desti)
# for curr_share in TICKERSLST_USE: myHELP.yf_downloadSingleData(curr_share, VS_TS, EF_TS, 3, "")
# 'ADBE', 'ABNB', 'GOOGL'

SHARESLST = TICKERSLST_USE2
print(SHARESLST)

# select_method = "MVO", "LGBM", "ARIMA"
def result_(select_method):
    notFirst = False
    for currShareName in SHARESLST:
        currShare = Share()
        currShare.setupShare(currShareName, "return", True, desti)
        currShare.switchLst = SWITCHLST_F
        # currShare.display()
        # for df_name in currShare.split_set.keys(): print(currShare.split_set[df_name])
        currShare.save()

        train_Columns = list(currShare.split_set["train_x"].columns).copy()
        train_Columns.remove("Y1")
        # train_Columns.remove("return")
        predicted_df = pd.DataFrame()

        # StrategyPart - LGBM
        if (select_method == "LGBM"):
            predicted_df = myTrainer.ts_predict_fn(currShare.split_set, train_Columns)
            predicted_df['ticker'] = currShare.stock_name
            predicted_df['pTradeRate'] = predicted_df.apply(lambda row: (row['Y1'] if row['predicted_Y1'] > 0 else 0), axis=1)
            predicted_df["tTradeBool"] = predicted_df.apply(lambda row: (row['Y1'] if row['Y1'] > 0 else 0), axis=1)
            print(predicted_df)

        # StrategyPart - ARIMA
        elif (select_method == "ARIMA"):
            predicted_df = myTrainer.arima_predict_fn(currShare.split_set, train_Columns, (1, 1, 1), (1, 1, 1, 12))
            predicted_df['ticker'] = currShare.stock_name
            # predicted_df['pTradeRate'] = predicted_df.apply(lambda row: (row['Y1'] if row['predicted_Y1'] > 0 else 0), axis=1)
            # predicted_df["tTradeBool"] = predicted_df.apply(lambda row: (row['Y1'] if row['Y1'] > 0 else 0), axis=1)
        
        # StrategyPart - MOV
        elif (select_method == "MOV"):
            # predicted_df['ticker'] = currShare.stock_name
            pass

        # Save and append the result to the result we want to have.
        # myHELP.save_file(currShareName, predicted_df, "", "predicted_")
        if (notFirst):
            predicted_df.to_csv(f'Result_{select_method}.csv', mode='a', index=True, header=False)
        else:
            predicted_df.to_csv(f'Result_{select_method}.csv', mode='w', index=True, header=True)
            notFirst = True

result_("LGBM")
result_("ARIMA")



    
    
