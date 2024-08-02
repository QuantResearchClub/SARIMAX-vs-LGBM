import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from   sklearn.model_selection import train_test_split, GridSearchCV
from   sklearn import preprocessing, metrics
from   sklearn.preprocessing import StandardScaler
from   sklearn.inspection import permutation_importance
from   datetime import date
import os

from Info import *

import warnings
warnings.filterwarnings("ignore")

"""
class of help functions:
in this case we downlaod the data and store them in separate folders, named in shares.
"""
class HelpFunctions():
    def __init__(self) -> None:
        pass

    # Download the data of a particular dataclass.
    # Class includes: Open, High, Low, Close, Volume.
    # This could be used as the
    def yf_downloadDataClass(self, tickersLst, s_timestamp: str, f_timestamp: str, dataclass: str, digits: int, desti: str):
        tickersDict = dict(zip(tickersLst, list(digits for i in range(len(tickersLst)))))
        data = yf.download(tickersLst, s_timestamp, f_timestamp, auto_adjust=True)[dataclass]
        filename = "_".join([s_timestamp, f_timestamp, dataclass])
        data = data.round(tickersDict)
        
        # data.to_csv(f"Dataset\\{desti}\\{filename}.csv")
        self.save_file(filename, data, desti, "source")
        return data

    # Prepare Trading data for a single share.
    def yf_downloadData(self, s_timestamp: str, f_timestamp: str, digits: int, desti: str):       
        # Step 01: Read and print the stock tickers that make up Nasdaq 100 indexes.
        print(f"DownloadData: {s_timestamp} to {f_timestamp}.\n")
        tickerLst = TICKERSLST.copy()

        # Step 02: Download the data for all tickers.
        for currTicker in tickerLst:
            self.yf_downloadSingleData(currTicker, s_timestamp, f_timestamp, digits, desti)
            
    # Function of download a single ticker for price prediction.
    def yf_downloadSingleData(self, currTicker: str, s_timestamp: str, f_timestamp: str, digits: int, desti: str):
        try:
            print(f"Download {currTicker}, {s_timestamp} to {f_timestamp}")
            data = yf.download([currTicker], s_timestamp, f_timestamp, auto_adjust=True)
            data = data.round({currTicker: digits})
            self.save_file(currTicker, data, desti, "source")

        except Exception as err:
            print(f"Download Exception: {currTicker}, {s_timestamp} to {f_timestamp}.\n {err}")  

    # Save data-file downloaded.
    def save_file(self, currTicker: str, data: pd.DataFrame, desti: str, prefix: str):
        c = 1
        curr_path = ["Dataset", currTicker, desti] if (desti != "") else ["Dataset", currTicker]
        str_currPath = "\\".join(curr_path)
        while (c <= len(curr_path)):
            try: os.mkdir("/".join(curr_path[:c]))
            except: pass
            c = c + 1
        data.to_csv(f"{str_currPath}\\{prefix}_{currTicker}.csv")

class Share():
    def __init__(self):
        self.stock_name = ""
        self.predicted_target = ""
        self.src_data = pd.DataFrame()
        self.run_data = pd.DataFrame()
        self.split_set = dict()
        self.switchLst = SWITCHLST
        self.HELP = HelpFunctions()
    
    # Set-up the basic information of a share data.
    def setupShare(self, share_name: str, target: str, run: bool, desti: str):
        self.stock_name = share_name
        self.predicted_target = target
        loc = LOCATION + f"/Dataset/{share_name}"
        loc = loc + f"/{desti}" if (desti != "") else loc

        try:
            try:
                src_df = pd.read_csv(loc + f"/source_{share_name}.csv")
            except Exception as err:
                print(f"Could not find the source dataset of {share_name}.\n Exception: {err}")
            
            self.src_data = src_df.copy()
            self.run_data = self.src_data.copy().dropna()

            if (run): 
                self.runShareData()
                self.set_predicted_vec(self.predicted_target)
                self.run_data = self.run_data.dropna() 
            
            print(f"Complete {self.stock_name}")
            self.split_set = self.train_test_split_by_date(self.run_data, SPLIT_DATE)
        
        except Exception as err:
            print(f"Failed in run pre-calculations of {share_name}.\n Exception: {err}")
    
    # Func to run the share data.
    def runShareData(self):
        try:
            # self.index_to_datetime()
            if self.switchLst["STD"]: self.std()
            if self.switchLst["BOLLINGER"]: self.bollinger_strat(10, 2)
            if self.switchLst["HA"]: self.heikin_ashi()
            if self.switchLst["SO"]: self.stochastic_oscillator(10, 3)
            if self.switchLst["BBR"]: self.bears_bulls_power(10)
            if self.switchLst["RSI"]: self.rsi(10)
            if self.switchLst["LOGRET"]: self.logret()
            if self.switchLst["MOM"]: self.mom(10)
            if self.switchLst["OBV"]: self.obv()
            if self.switchLst["ADR"]: self.accum_distribute()
        except Exception as err:
            print(f"Calculation Failure: {self.stock_name}.\nException: {err}")
    
    # Function to calculate the predicted vector after "window" trading day.
    # Y1 represents predict the target with 1 day later, 'ret_bin'
    # Y2 represents predict the target with next_n days
    def set_predicted_vec(self, target: str):
        self.run_data['Y1'] = self.run_data[target].shift(-1)
        self.run_data['Y2'] = np.where(self.run_data[target].rolling(1).sum() >= self.run_data[target].median(), 1,0)
        self.run_data['Y2'] = self.run_data['Y2'].shift(-1)
    
    # Function to seperate the data into training set and validation set.
    def train_test_split_by_date(self, ipt_data: pd.DataFrame, split_date):
        # Set date as index
        data = ipt_data.copy()
        if (DATEVEC in list(data.columns)): 
            data[DATEVEC] = pd.to_datetime(data[DATEVEC])
            data.set_index([DATEVEC], inplace = True)
        else:
            print("Datavec is not in the columns of current dataframe.")
            return ipt_data
        
        if (type(split_date) == str): split_date = pd.to_datetime(split_date)
        # Store the sets in a dictionary under train_x, train_y and testing set.
        sets = {
            'train_x': data[data.index <= split_date].sort_index(),
            'train_y': data[data.index <= split_date].sort_index()[self.predicted_target],
            'test':    data[data.index > split_date].sort_index()
        }
        return sets
    
    def display(self):
        print(f"{FENGEXIAN}\nCurrent Srock: {self.stock_name}")
        for set_name in self.split_set.keys():
            print(f"\n{set_name}:")
            print(self.split_set[set_name])
    
    def save(self):
        self.HELP.save_file(self.stock_name, self.run_data, "", "Run")
        for prefixName in self.split_set.keys():
            try: self.HELP.save_file(self.stock_name, self.split_set[prefixName], "", prefixName)
            except: pass
        
    # Attribute Functions:
    ######################################################################################################################
    # There are 21 trading day in a month, 
    # sd_m counts the std of 1 month after starting day by rolling
    # sd_w counts the std of 1 week  after starting day by rolling
    def std(self):
        self.run_data['sd_m'] = self.run_data['Close'].rolling(21).std()
        self.run_data['sd_w'] = self.run_data['Close'].rolling(5).std()

    # Calculate the bollinger_strat
    # - bollinger_strat is calculated by the close price.
    def bollinger_strat(self, window, no_of_std):
        rolling_mean = self.run_data['Close'].rolling(window).mean()
        rolling_std = self.run_data['Close'].rolling(window).std()
        self.run_data['boll_high'] = rolling_mean + (rolling_std * no_of_std)
        self.run_data['boll_low'] = rolling_mean - (rolling_std * no_of_std)     

    # Calculate the heikin_ashi
    # - heikin_ashi open_HA is calculated as the average of open price and close price of the previous date.
    # - hiekin_ashi close_HA is calculated as the average price of current day's ("openHA", "Close", "High", "Low").
    def heikin_ashi(self):
        self.run_data['open_HA']  = (self.run_data['Open'].shift(1) + self.run_data['Close'].shift(1))/2
        self.run_data['close_HA'] = (self.run_data['open_HA']+self.run_data['Close']+self.run_data['High']+self.run_data['Low'])/4
    
    # Calculate the log return. (Rate of the next day comparing with today)
    # median works better for returns and trading
    def logret(self):
        self.run_data['return']  = np.log(1 + self.run_data['Close'].pct_change())
        self.run_data['ret_bin'] = np.where(self.run_data['return'] >= self.run_data['return'].median(), 1, 0) 
    
    # Calculate the Stochastic Oscillator
    def stochastic_oscillator(self, n, m):
        high_n = self.run_data['High'].rolling(window=n).max()
        low_n = self.run_data['Low'].rolling(window=n).min()
        self.run_data['k'] = ((self.run_data['Close'] - low_n) / (high_n - low_n)) * 100
        self.run_data['d'] = self.run_data['k'].rolling(window = m).mean()
        self.run_data['sto_oscil'] = self.run_data['k'] - self.run_data['d']
        self.run_data.drop(columns=['k', 'd'], inplace=True)
        #return k, d

    # Calculate the bears_bulls_power
    # bears_bull_power 参数
    def bears_bulls_power(self, ma_window):
        ma = self.run_data['Close'].rolling(window=ma_window).mean()
        self.run_data['bull_power'] = self.run_data['High'] - ma
        self.run_data['bear_power'] = self.run_data['Low'] - ma
        #return bears_power, bulls_power

    # rsi value
    def rsi(self, window):
        delta = self.run_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        self.run_data['rsi'] = 100 - (100 / (1 + rs))
        #return rsi

    # mom_value
    def mom(self, window):
        self.run_data['mom'] = self.run_data['Close'].rolling(window).apply(np.sum)

    # obv_value
    def obv(self):
        self.run_data['obv1'] = np.where(self.run_data['Close'] >= self.run_data['Close'].shift(-1),1,-1)*self.run_data['Volume']
        self.run_data['obv'] = self.run_data['obv1'].cumsum()
        self.run_data.drop(columns=['obv1'], inplace=True)

    # accum_distribution
    def accum_distribute(self):
        self.run_data['ad1'] = self.run_data['Volume']*(2*self.run_data['Close']-self.run_data['Low']-self.run_data['High'])/(self.run_data['High']-self.run_data['Low'])
        self.run_data['acc_dis'] = self.run_data['ad1'].cumsum()
        self.run_data.drop(columns=['ad1'], inplace=True)

    # aroon
    def aroon(self, window):
        self.run_data['up'] = 100 * self.run_data["High"].rolling(window).apply(lambda x: x.argmax()) / window
        self.run_data['dn'] = 100 * self.run_data["Low"].rolling(window).apply(lambda x: x.argmin()) / window
        self.run_data['aroon'] = self.run_data['up'] - self.run_data['dn'] 

    # atr
    def atr(self, window):
        high_low = self.run_data['High'] - self.run_data['Low']
        high_close = np.abs(self.run_data['High'] - self.run_data['Close'].shift())
        low_close = np.abs(self.run_data['Low'] - self.run_data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.run_data['atr'] = true_range.rolling(window).sum()/window

class Portofolio():
    # Initial function: dataframe, indexed
    def __init__(self):
        self.method = ""
        self.data       = pd.DataFrame()
        self.tickersLst = []
        self.p_df_1 = pd.DataFrame()
        self.p_df_2 = pd.DataFrame()
        self.p_df_3 = pd.DataFrame()
        self.p_df_dict = {}
    
    def set_up(self, ipt_df: pd.DataFrame, method: str):
        self.method     = method
        self.data       = ipt_df.copy()
        self.tickersLst = list(self.data['ticker'].unique())
        self.p_df_dict  = dict(zip(self.tickersLst, list(0 for i in range(len(self.tickersLst)))))
    
    def setPortofolioDf(self):
        p_df = pd.DataFrame(list(self.share_dict for i in range(len(self.date_list))))
        p_df['Date'] = self.date_list
        p_df.set_index(p_df['Date'], inplace = True)
        return p_df[self.tickersLst]

    # The trading strategy function:
    # - pred_vec represents the column name of the trading decision vector. (could be "Y1" or "predicted_Y1")
    # - n means we choose the top 5 shares for the prediction using.
    def trading(self, data: pd.DataFrame, pred_vec: str, n = 5):
        # top: True value if 
        # - the predicted result is greater than 0.
        # - it is of the top 5 value on the day.
        data['top'] = data.groupby('Date')[pred_vec].rank(method='first', ascending=False) <= n
        data['top'] = data.apply(
            lambda row: True if (row['top'] and row[pred_vec] > 0) else False, 
            axis = 1
        )
        data['top'] = data['top'].astype(int)

        # Assign the predicted set.
        dateLst = list(data["Date"].unique())
        z = []
        for currDate in dateLst:
            try:
                temp_df = data[data['Date'] == currDate].copy()
                curr_p_df_dict = self.p_df_dict.copy()
                for ticker in self.tickersLst:
                    temp_df_1 = temp_df[temp_df['ticker'] == ticker].reset_index(drop = True)
                    curr_p_df_dict[ticker] = temp_df_1['top'][0]
                z.append(curr_p_df_dict)
            except: 
                pass
        self.p_df_3 = pd.DataFrame(z)
        
        # calculate return if it is in the list.
        x = []
        for ticker in self.tickersLst:
            tt = data[data['ticker'] == ticker]
            tt['strat'] = tt['return'] * tt['top'].shift(1)
            x.append(tt)
        x = pd.concat(x)
        x = x.dropna()
        self.p_df_1 = x.copy()

        # calculate the detailed dataframe.
        t = []
        for date in x['Date'].unique():
            tt = x[x['Date'] == date]
            trade = tt['strat'].sum()/n
            ret = tt['return'].mean()
            t.append(pd.DataFrame({'date': [date], 'return': [ret], 'trade': [trade],}))

        t = pd.concat(t)
        t.set_index('date', inplace=True)
        self.p_df_2 = t.copy()
        return t
    
