import pandas as pd
import os

# Date Timestamp
TS_TS = '2023-1-1' 
TF_TS = '2023-12-31'
VS_TS = '2024-1-1'
VF_TS = '2024-6-30'
ES_TS = '2024-7-1'
EF_TS = '2024-8-1'

# Key split date.
SPLIT_DATE = '2024-6-30'

# 1. Absolute Location
LOCATION = "C:/Users/liucu/Developments/Python_Project/DSA5205PORTFOLIO"
# Reading format: pd.read_csv(LOCATION + f"/{share_name}/source_{share_name}.csv")

# 2. Alternating the factor list. Do not change the structure, just modify the value only.
SWITCHLST = {
    "STD": True,
    "BOLLINGER": True,
    "HA":  True,
    "SO":  True,
    "BBR": True,
    "RSI": True,
    "LOGRET": True,
    "MOM": True,
    "OBV": True,
    "ADR": True
}

SWITCHLST_F = {
    "STD": False,
    "BOLLINGER": True,
    "HA":  True,
    "SO":  False,
    "BBR": False,
    "RSI": False,
    "LOGRET": True,
    "MOM": False,
    "OBV": False,
    "ADR": False
}

# TimeSeries Paras
TS_LGBM_PARAMS = {"boosting_type": "gbdt", "metric": 'binary_logloss', "verbosity": -1, 'random_state': 13}

# We picked 10 top
TICKERSLST_USE1 = ['AAPL','MSFT','NVDA','GOOG','AMZN','META','TSM','LLY','TSLA','AVGO']
TICKERSLST_USE2 = ['WBD', 'ODFL', 'CMCSA', 'PYPL', 'KHC', 'GILD', 'GEHC', 'FAST', 'BKR', 'ADP']
TIME_LAPSE = 21

"""
Do not modify the below part:
"""
# TICKERS = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#ComponenTS')[4]
# TICKERSLST = TICKERS.Ticker.to_list()
TICKERSLST = os.listdir("Dataset")
FENGEXIAN = "".join(['-'] * 30)
DATEVEC = "Date"


