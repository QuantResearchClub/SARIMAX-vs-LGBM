import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from datetime import date

# Download the data of a particular dataclass.
# Open, High, Low, Close, Volume
def yf_downloadDataClass(tickersLst, s_timestamp: str, f_timestamp: str, dataclass: str, digits: int):
    tickersDict = dict(zip(tickersLst, list(digits for i in range(len(tickersLst)))))
    data = yf.download(tickersLst, s_timestamp, f_timestamp, auto_adjust=True)[dataclass]
    filename = "_".join([s_timestamp, f_timestamp, dataclass])
    data = data.round(tickersDict)
    data.to_csv(f"Dataset\\{filename}.csv")
    return data

# Prepare data.
"""
    # Training Set
    s_timestamp = '2024-1-1'
    f_timestamp = '2024-6-30'
    yf_downloadDataClass(s_timestamp, f_timestamp, 2)

    # Validation Set
    s_timestamp = '2024-7-1'
    f_timestamp = '2024-7-31'
    yf_downloadDataClass(s_timestamp, f_timestamp, 2)
"""
def yf_downloadData(s_timestamp: str, f_timestamp: str, digits: int):
    # Read and print the stock tickers that make up Nasdaq 100 indexes.
    tickers = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Components')[4]
    tickerLst = tickers.Ticker.to_list()

    # Get the data for this tickers from yahoo finance
    for currClass in ["Open", "High", "Low", "Close", "Volume"]:
        yf_downloadDataClass(
            tickersLst = tickerLst,
            s_timestamp = s_timestamp,
            f_timestamp = f_timestamp,
            dataclass = currClass,
            digits = digits
        )

class sharesDS():
    def __init__(self):
        self.dataset = pd.DataFrame()

    def setup_dataset(self):
        return

