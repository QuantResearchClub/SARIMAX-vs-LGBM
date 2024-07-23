import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from datetime import date
import os



class Kernel():
    def __init__(self):
        pass

    # Download the data of a particular dataclass.
    # Open, High, Low, Close, Volume
    def yf_downloadDataClass(self, tickersLst, s_timestamp: str, f_timestamp: str, dataclass: str, digits: int, desti: str):
        tickersDict = dict(zip(tickersLst, list(digits for i in range(len(tickersLst)))))
        data = yf.download(tickersLst, s_timestamp, f_timestamp, auto_adjust=True)[dataclass]
        filename = "_".join([s_timestamp, f_timestamp, dataclass])
        data = data.round(tickersDict)
        data.to_csv(f"Dataset\\{desti}\\{filename}.csv")
        return data

    # Prepare data.
    def yf_downloadData(self, s_timestamp: str, f_timestamp: str, digits: int, desti: str):
        print(f"DownloadData: {s_timestamp} to {f_timestamp}.\n")
        # Read and print the stock tickers that make up Nasdaq 100 indexes.
        tickers = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Components')[4]
        tickerLst = tickers.Ticker.to_list()

        for currTicker in tickerLst:
            try:
                print(f"Download {currTicker}, {s_timestamp} to {f_timestamp}")
                data = yf.download([currTicker], s_timestamp, f_timestamp, auto_adjust=True)
                filename = "_".join([s_timestamp, f_timestamp, currTicker])
                data = data.round({currTicker: digits})

                curr_path = ["Dataset", currTicker, desti]
                c = 1
                while (c <= len(curr_path)):
                    try: os.mkdir("\\".join(curr_path[:c]))
                    except: pass
                    c = c + 1
                data.to_csv(f"Dataset\\{currTicker}\\{desti}\\{filename}.csv")
            except Exception as err:
                print(f"Download Exception: {currTicker}, {s_timestamp} to {f_timestamp}.\n {err}")

class TradeData():
    def __init__(self):
        self.start_date = ""
        self.end_date = ""
        self.stock = ""
        self.data = pd.DataFrame()
    
    def setup(self, dataset, s_ts, f_ts, sharename):
        self.data = dataset.copy()
        self.start_date = s_ts
        self.end_date = f_ts
        self.stock = sharename
        
    def to_datetime(self, date_vecname: str):
        datas = pd.to_datetime(self.data[date_vecname], format = "%Y-%m-%d")
        self.data = self.data.set_index(datas)
        
    # There are 21 trading day in a month, 
    # sd_m counts the std of 1 month after starting day by rolling
    # sd_w counts the std of 1 week  after starting day by rolling
    # ---------------------------------------------------------------------------------------------
    def std(self):
        self.data['sd_m'] = self.data['Close'].rolling(21).std()
        self.data['sd_w'] = self.data['Close'].rolling(5).std()

    # Do the bollinger_strat
    def bollinger_strat(self, window, no_of_std):
        rolling_mean = self.data['Close'].rolling(window).mean()
        rolling_std = self.data['Close'].rolling(window).std()
        self.data['boll_high'] = rolling_mean + (rolling_std * no_of_std)
        self.data['boll_low'] = rolling_mean - (rolling_std * no_of_std)     

    def heikin_ashi(self):
        self.data['open_HA'] = (self.data['Open'].shift(1) + self.data['Close'].shift(1))/2
        self.data['close_HA'] = (self.data['open_HA']+self.data['Close']+self.data['High']+self.data['Low'])/4
    
    def logret(self):
        data = self.data
        data['return'] = np.log(1+ data['Close'].pct_change())
        data['ret_bin'] = np.where(data['return'] >= data['return'].median(), 1,0) # median works better for returns and trading
        self.data = data.copy()

    """
    def stochastic_oscillator(self, data, n, m):
        high_n = self.data['High'].rolling(window=n).max()
        low_n = self.data['Low'].rolling(window=n).min()
        self.data['k'] = ((self.data['Close'] - low_n) / (high_n - low_n)) * 100
        self.data['d'] = self.data['k'].rolling(window=m).mean()
        self.data['sto_oscil'] = self.data['k'] - self.data['d']
        data.drop(columns=['k', 'd'], inplace=True)
        #return k, d

    def bears_bulls_power(self, data, ma_window):
        ma = self.data['Close'].rolling(window=ma_window).mean()
        self.data['bull_power'] = self.data['High'] - ma
        self.data['bear_power'] = self.data['Low'] - ma
        #return bears_power, bulls_power

    def rsi(self, data, window):
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        #return rsi

    def logret(self, data):
        self.data['return'] = np.log(1+ self.data['Close'].pct_change())
        self.data['ret_bin'] = np.where(self.data['return'] >= self.data['return'].median(), 1,0) # median works better for returns and trading
        self.data['Y1'] = self.data['ret_bin'].shift(-1)
        self.data['Y2'] = np.where(self.data['return'].rolling(n).sum() >= self.data['return'].median(), 1,0)
        self.data['Y2'] = self.data['Y2'].shift(-(n))
        #return data

    def mom(self, data, window):
        self.data['mom'] = self.data['Close'].rolling(window).apply(np.sum)

    def obv(data):
        self.data['obv1'] = np.where(self.data['Close'] >= self.data['Close'].shift(-1),1,-1)*self.data['Volume']
        self.data['obv'] = self.data['obv1'].cumsum()
        data.drop(columns=['obv1'], inplace=True)

    def accum_distribute(self, data):
        self.data['ad1'] = self.data['Volume']*(2*self.data['Close']-self.data['Low']-self.data['High'])/(self.data['High']-self.data['Low'])
        self.data['acc_dis'] = self.data['ad1'].cumsum()
        data.drop(columns=['ad1'], inplace=True)

    def aroon(self, data, window):
        self.data['up'] = 100 * data.High.rolling(window).apply(lambda x: x.argmax()) / window
        self.data['dn'] = 100 * data.Low.rolling(window).apply(lambda x: x.argmin()) / window
        self.data['aroon'] = self.data['up'] - self.data['dn'] 

    def atr(self, data, window):
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.data['atr'] = true_range.rolling(window).sum()/window
    """