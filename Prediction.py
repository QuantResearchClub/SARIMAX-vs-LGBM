import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from datetime import date
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from Info import *


class TimeSeriesTraining():
    def __init__(self, params = TS_LGBM_PARAMS, n_f = 3) -> None:
        self.n_fold = n_f
        self.folds = self.create_folds(self.n_fold)
        self.params = params
        self.classifier = {}
        self.mse = 0
    
    # Split the dataset into k_folds, and taking validation cross.
    def create_folds(self, n_fold = 3):
        folds = TimeSeriesSplit(
            n_splits = n_fold,
            gap = 0,                  # gap between the train and test side of the splits. 
            max_train_size = 10000,
            test_size = 10,           # about 10% of training data
        )
        return folds

    # Return the predicted dataframe basing on "Y1"
    # target selection range ("Y1" or "Y2"), or other target, it is set flexible.
    def ts_predict_fn(self, df_split: dict, ipt_columns: list, target = "Y1"):
        splits = self.folds.split(df_split['train_x'], df_split['train_y'])
        y_preds = np.zeros(df_split['test'].shape[0])
        y_oof   = np.zeros(df_split['train_x'].shape[0])

        feature_importances = pd.DataFrame()
        feature_importances['feature'] = ipt_columns

        for fold_n, (train_index, valid_index) in enumerate(splits):
            print('Currnet Fold:',fold_n + 1)
            X_train = df_split['train_x'][ipt_columns].iloc[train_index]
            X_valid = df_split['train_x'][ipt_columns].iloc[valid_index]
            y_train = df_split['train_y'].iloc[train_index]
            y_valid = df_split['train_y'].iloc[valid_index]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid)
            clf = lgb.train(self.params, dtrain, 50, valid_sets = [dtrain, dvalid])
            self.classifier = clf

            num_iteration=clf.best_iteration
            # feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
            y_pred_valid = clf.predict(X_valid,num_iteration=clf.best_iteration)
            y_oof[valid_index] = y_pred_valid
            y_preds += clf.predict(df_split['test'][ipt_columns], num_iteration=num_iteration) / self.n_fold
        
        predicted_vecName = f'predicted_{target}'
        df_split['test'][predicted_vecName] = y_preds
        df_split['test']['binary'] = np.where(y_preds >= 0.5, 1,0)
        df_predicted = df_split['test'][['return', target, predicted_vecName, 'binary']]
        # self.mse = mean_squared_error(df_predicted, test_pred)

        return df_predicted
    
    # Arima prediction strategy.
    def arima_predict_fn(self, df_split: dict, ipt_column: list, pdq, s_order):
        model = sm.tsa.statespace.SARIMAX(
            endog = df_split["train_y"], 
            exog  = df_split["train_x"][ipt_column],
            order = pdq,
            seasonal_order = s_order
        )
        arima_result = model.fit()
        # print(arima_result.summary())
        test_pred = arima_result.forecast(
            steps = df_split['test'].shape[0],
            exog  = df_split['test'][ipt_column]
        )
        print(test_pred)
        
        test_pred.index = df_split['test'].index
        resultDF = df_split['test'][["return", "Y1"]].copy()
        resultDF = resultDF.join(test_pred)
        resultDF['binary'] = np.where(resultDF["predicted_mean"] >= 0.5, 1, 0)
        # self.mse = mean_squared_error(test_pred, test_pred)
        return resultDF
    
    # MOV prediction strategy.
    def mov_predict_fn(self, df_split: dict, ipt_column: list):
        train = df_split["train_x"]["return"]
        test  = df_split["test"]["return"]
        
        mu = mean_historical_return(train)
        S = CovarianceShrinkage(train).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        
        weights = ef.max_sharpe()
        weights = ef.clean_weights()

        latest_prices = get_latest_prices(train)
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=1000000)
        allocation, leftover = da.lp_portfolio()
        self.mvo_rough_weight_ret(test, weights)
        self.mvo_discrete_ret(test, allocation)
        print(latest_prices)
        return latest_prices
    
    def mvo_rough_weight_ret(self, data, weights):
        # Make a copy of the original data
        data_copy = data.copy()
        
        # calculate returns of test portfolio, using rough weights
        for ticker, weight in weights.items():
            if ticker in data_copy.columns:
                data_copy[ticker] = data_copy[ticker] * weight
        data_copy['returns'] = data_copy.sum(axis=1)
        
        # convert to log returns
        data_copy['returns'] = np.log( 1 + data_copy['returns'])

        # calculate returns over 1 year test period
        annual_ret = np.exp(data_copy['returns'].sum())-1
        print('------using rough weights------')
        print('Annual regular return: ',annual_ret)

        # calculate standard deviation over same period
        annual_std = (np.exp(data_copy['returns'])-1).std()
        print('Annual regular std: ', annual_std)

        # calculate sharpe ratio
        sr = annual_ret/ annual_std
        print("Sharpe Ratio is: ", sr)
        print('--------------------------------')
        data_copy[["returns"]].cumsum().plot()
    
    def mvo_discrete_ret(self, data, allocation):
        # Make a copy of the original data
        data_copy = data.copy()
        
        # calculate allocation of test portfolio, using discrete allocation
        for column in data.columns:
            if column in allocation:
                data_copy[column] = data_copy[column]*allocation[column]
            else:
                data_copy[column] = data_copy[column]*0

        # calculate portfolio size
        data_copy['balance'] = data_copy.sum(axis=1)

        # calculate portfolio (log) returns
        data_copy['ret'] = np.log ( 1 + data_copy['balance'].pct_change())

        # calculate returns over 1 year test period
        annual_ret = np.exp(data_copy['ret'].sum())-1
        print('------using discrete allocation------')
        print('Annual regular return: ',annual_ret)

        # calculate standard deviation over same period
        annual_std = (np.exp(data_copy['ret'])-1).std()
        print('Annual regular std: ', annual_std)

        # calculate sharpe ratio
        sr = annual_ret/ annual_std
        print("Sharpe Ratio is: ", sr)
        print('--------------------------------')
        
        data_copy[["ret"]].cumsum().plot()
    





"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing, metrics
from datetime import date
from scipy.stats import ttest_rel, t
from scipy import stats

tickers = ['AAPL','MSFT','NVDA','GOOG','AMZN','META','TSM','LLY','TSLA','AVGO']

cwd = 'C:\\Users\\dawei\\Dropbox\\NUS\\DSA5205\\project\\'

# include all data
lgbm = pd.read_csv(cwd+'LGBM.csv')
lgbm['Date'] = pd.to_datetime(lgbm['Date'])
mvo = pd.read_csv(cwd+'MVO\\MVO.csv')
mvo['date'] = pd.to_datetime(mvo['Date'])
mvo.set_index('date', inplace=True)
arima = pd.read_csv(cwd+'ARIMA_Model\\ARIMA_result_return.csv')
arima = arima.drop(arima[~arima['Share'].isin(tickers)].index)
arima['Date'] = pd.to_datetime(arima['Date'], dayfirst=True)
arima.rename(columns={'predicted_mean': 'predicted', 'Share': 'ticker'}, inplace=True)

# create ensemble strategy through MinMax scaling
comb = pd.merge(arima, lgbm, on=['ticker','Date'], how = 'left')
comb['x'] = (comb['predicted_x'] - comb['predicted_x'].min())/(comb['predicted_x'].max() - comb['predicted_x'].min())
comb['y'] = (comb['predicted_y'] - comb['predicted_y'].min())/(comb['predicted_y'].max() - comb['predicted_y'].min())
comb['predicted'] = (comb['x']+comb['y'])/2
comb = comb[['Date','ticker','predicted','return_y']]
comb.rename(columns={'return_y': 'return'}, inplace=True)

n = 5 # number of firms included in portfolio

# apply trading strategy
def trading(data):
    data['top'] = data.groupby('Date')['predicted'].rank(method='first', ascending=False) <= n
    data['top'] = data['top'].astype(int)

    # calculate returns
    x = []
    for ticker in data['ticker'].unique():
        tt = data[data['ticker'] == ticker]
        tt['strat'] = tt['return']*tt['top'].shift(1)
        x.append(tt)
    x = pd.concat(x)
    x = x.dropna()

    t = []
    for date in x['Date'].unique():
        tt = x[x['Date'] == date]
        trade = tt['strat'].sum()/n
        ret = tt['return'].mean()
        t.append(pd.DataFrame({'date': [date], 'return': [ret], 'trade': [trade],}))
    t = pd.concat(t)
    t.set_index('date', inplace=True)
    return t

lgbm_ = trading(lgbm)
arima_ = trading(arima)
comb_ = trading(comb)

# plot everything
lgbm_['lgbm'] = lgbm_['trade']
mvo['mvo'] = mvo['trade']
arima_['arima'] = arima_['trade']
comb_['ensemble'] = comb_['trade']
df = pd.concat([lgbm_['lgbm'], mvo['mvo'], arima_['arima'], comb_['ensemble'], lgbm_['return']], axis=1).dropna()

plt.figure(figsize=(10, 3))
plt.plot(df['return'].cumsum(), label='benchmark')
plt.plot(df['lgbm'].cumsum(), label='LGBM')
plt.plot(df['arima'].cumsum(), label='ARIMA')
plt.plot(df['mvo'].cumsum(), label='MVO')
plt.plot(df['ensemble'].cumsum(), label='ensemble')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns Over Time')
plt.legend()
plt.grid(False)
plt.show()

# correlation
correlation_matrix = df.corr()
print(correlation_matrix)
#plt.figure(figsize=(12, 8))
#sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
#plt.title('Correlation Heatmap of Selected Columns')
#plt.show()

# sharpe ratio, returns, std
def sharpe_ratio(data):
    data_copy = data.copy()

    # calculate returns over test period
    annual_ret = np.exp(data_copy.sum())-1
    print('--------------------------------')
    print('regular return: ',annual_ret)

    # calculate standard deviation over same period
    annual_std = (np.exp(data_copy)-1).std()
    print('regular std: ', annual_std)

    # calculate sharpe ratio
    sr = annual_ret/ annual_std
    print("Sharpe Ratio is: ", sr)
    print('--------------------------------')

sharpe_ratio(df['lgbm'])
sharpe_ratio(df['mvo'])
sharpe_ratio(df['arima'])
sharpe_ratio(df['ensemble'])
sharpe_ratio(df['return'])

# drawdown
def plot_drawdown(data):
    cumulative_returns = pd.DataFrame()
    cumulative_returns['trade'] = data.cumsum()
    cumulative_returns["Cum_Max_ret"] = cumulative_returns["trade"].cummax()
    
    cumulative_returns['drawdown']=cumulative_returns["Cum_Max_ret"]-cumulative_returns["trade"]
    print("max drawdown:",cumulative_returns['drawdown'].max())
    plt.figure(figsize=(10, 3))
    plt.plot(cumulative_returns['trade'], label='')
    plt.plot(cumulative_returns['Cum_Max_ret'], label='')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Drawdown Over Time')
    plt.legend()
    plt.grid(False)
    plt.show()

plot_drawdown(df['lgbm'])
plot_drawdown(df['mvo'])
plot_drawdown(df['arima'])
plot_drawdown(df['ensemble'])
plot_drawdown(df['return'])

# statistical significance
results = {}
for model in ['lgbm', 'mvo', 'arima', 'ensemble']:
    t_stat, p_value = ttest_rel(df[model], df['return'])
    results[model] = {'t_stat': t_stat, 'p_value': p_value}

results_df = pd.DataFrame(results).T
print(results_df)

# Plot Q-Q plots to compare 
plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns, 1):
    plt.subplot(2, 3, i)
    (osm, osr), (slope, intercept, r) = stats.probplot(df[column].dropna(), dist="t", sparams=(len(df)-1,))
    plt.scatter(osm, osr*100, color='blue', alpha=0.5, label='train')
    min_val = min(osm.min(), osr.min())
    max_val = max(osm.max(), osr.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    plt.title(f'{column}')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
   # plt.legend()
plt.tight_layout()
plt.show()

# VaR and ES
confidence_level = 0.9

def calculate_kde_es(data, confidence_level):
    # Calculate Expected Shortfall (ES) using KDE. 
    kde = stats.gaussian_kde(data)
    var = np.percentile(data, (1 - confidence_level) * 100)
    tail_values = data[data <= var]
    es = np.mean(tail_values)
    return var, es

es_results = {}

plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns, 1):
    data = df[column].dropna()
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)
    y = kde(x)

    plt.subplot(3, 2, i)
    plt.plot(x, y, label=f'KDE of {column}')
    plt.fill_between(x, y, where=(x <= np.percentile(data, (1 - confidence_level) * 100)), color='red', alpha=0.3)
    
    var, es = calculate_kde_es(data, confidence_level)
    es_results[column] = {'VaR': var, 'ES': es}
    
    plt.axvline(var, color='red', linestyle='--', label=f'VaR ({confidence_level*100:.1f}%)')
    plt.axvline(es, color='blue', linestyle='--', label=f'ES ({confidence_level*100:.1f}%)')
    plt.title(f'KDE and ES for {column}')
    plt.legend()
plt.tight_layout()
plt.show()

es_df = pd.DataFrame.from_dict(es_results, orient='index')
print(es_df)

#%% CAPM alpha and beta (ignore for now, not enough data)
ff5 = pd.read_csv('C:\\Users\\dawei\\Dropbox\\NUS\\DSA5205\\project\\F-F_Research_Data_5_Factors_2x3_daily.csv', skiprows=3)
ff5.drop(ff5.tail(62).index, inplace = True)
ff5.rename(columns={'Unnamed: 0':'date','Mkt-RF':'MKT'}, inplace=True)
ff5['date'] = pd.to_datetime(ff5['date'], format='%Y%m')
ff5 = ff5.assign(MKT = pd.to_numeric(ff5['MKT']), SMB = pd.to_numeric(ff5['SMB']), HML = pd.to_numeric(ff5['HML']),
                 RMW = pd.to_numeric(ff5['RMW']), CMA = pd.to_numeric(ff5['CMA']), RF = pd.to_numeric(ff5['RF']))

merged = pd.merge(df, ff5, on='date', how='left')
reg = smf.ols(formula = 'value_weighted_return ~ MKT + SMB + HML + RMW + CMA', data=merged).fit()
coefficients = reg.params
t_stats = reg.tvalues
results_df = pd.DataFrame({'Coefficient': coefficients, 'T-Statistic': t_stats})
"""
