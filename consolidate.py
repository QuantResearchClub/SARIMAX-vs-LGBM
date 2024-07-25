import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing, metrics
from datetime import date

tickers = ['AAPL','MSFT','NVDA','GOOG','AMZN','META','TSM','LLY','TSLA','AVGO']

#%% include all data
lgbm = pd.read_csv('C:\\Users\\dawei\\Dropbox\\NUS\\DSA5205\\project\\LGBM.csv')
lgbm['date'] = pd.to_datetime(lgbm['date'])
lgbm.set_index('date', inplace=True)
mvo = pd.read_csv('C:\\Users\\dawei\\Dropbox\\NUS\\DSA5205\\project\\MVO\\MVO.csv')
mvo['date'] = pd.to_datetime(mvo['Date'])
mvo.set_index('date', inplace=True)
arima = pd.read_csv('C:\\Users\\dawei\\Dropbox\\NUS\\DSA5205\\project\\ARIMA_Model\\ARIMA_result_return.csv')
arima = arima.drop(arima[~arima['Share'].isin(tickers)].index)
arima['Date'] = pd.to_datetime(arima['Date'], dayfirst=True)

# apply trading strategy to arima
arima['top'] = arima.groupby('Date')['predicted_mean'].rank(method='first', ascending=False) <= 5
arima['top'] = arima['top'].astype(int)

# calculate returns
x = []
for ticker in arima['Share'].unique():
    tt = arima[arima['Share'] == ticker]
    tt['strat'] = tt['return']*tt['top'].shift(1)
    x.append(tt)
x = pd.concat(x)
x = x.dropna()

t = []
for date in x['Date'].unique():
    tt = x[x['Date'] == date]
    trade = tt['strat'].sum()/5
    ret = tt['return'].mean()
    t.append(pd.DataFrame({'date': [date], 'return': [ret], 'trade': [trade],}))
t = pd.concat(t)
t.set_index('date', inplace=True)


# plot everything
lgbm['lgbm'] = lgbm['trade']
mvo['mvo'] = mvo['trade']
t['arima'] = t['trade']
df = pd.concat([lgbm['lgbm'], mvo['mvo'], t['arima'], lgbm['return']], axis=1).dropna()

plt.figure(figsize=(10, 3))
plt.plot(df['return'].cumsum(), label='benchmark')
plt.plot(df['lgbm'].cumsum(), label='LGBM')
plt.plot(df['mvo'].cumsum(), label='MVO')
plt.plot(df['arima'].cumsum(), label='ARIMA')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns Over Time')
plt.legend()
plt.grid(False)
plt.show()

# correlation
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Selected Columns')
plt.show()

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
plot_drawdown(df['return'])

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
