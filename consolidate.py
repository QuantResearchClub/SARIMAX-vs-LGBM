import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing, metrics
from datetime import date
from scipy.stats import ttest_rel, t
from scipy import stats
from Info import *
from Functions import *

# tickers = ['AAPL','MSFT','NVDA','GOOG','AMZN','META','TSM','LLY','TSLA','AVGO']
tickers = TICKERSLST_USE2

# cwd = 'C:\\Users\\dawei\\Dropbox\\NUS\\DSA5205\\project\\'
cwd = LOCATION + "\\DataResult\\"

# include all data
lgbm = pd.read_csv(cwd+'Result_LGBM.csv')
lgbm['Date'] = pd.to_datetime(lgbm['Date'])
mvo = pd.read_csv(cwd+'MVO.csv')
mvo['date'] = pd.to_datetime(mvo['Date'])
mvo.set_index('date', inplace=True)
arima = pd.read_csv(cwd+'Result_ARIMA.csv')
arima = arima.drop(arima[~arima['ticker'].isin(tickers)].index)
try: arima['Date'] = pd.to_datetime(arima['Date'], dayfirst=True) 
except: pass

# arima is x, lgbm is y
arima.rename(columns={'predicted_mean': 'predicted'}, inplace=True)
lgbm.rename(columns={'predicted_Y1': 'predicted'}, inplace=True)
arima["Date"] = pd.to_datetime(arima["Date"])
lgbm["Date"]  = pd.to_datetime(lgbm["Date"])

# create ensemble strategy through MinMax scaling
comb = pd.merge(arima, lgbm, on=['ticker', 'Date'], how = 'left')
comb['x'] = (comb['predicted_x'] - comb['predicted_x'].min())/(comb['predicted_x'].max() - comb['predicted_x'].min())
comb['y'] = (comb['predicted_y'] - comb['predicted_y'].min())/(comb['predicted_y'].max() - comb['predicted_y'].min())
comb['predicted'] = (comb['x']+comb['y'])/2
comb = comb[['Date','ticker','predicted','return_y']]
comb.rename(columns={'return_y': 'return'}, inplace=True)

# apply trading strategy
def trading(data, method):
    # number of firms included in portfolio
    n = 5 
    print(data.columns)
    myP= Portofolio()
    myP.set_up(ipt_df = data, method = method)
    # myP.trading(data, "Y1", n)
    myP.trading(data, "predicted", n)

    myP.p_df_1.to_csv(f"{method}_1.csv")
    myP.p_df_2.to_csv(f"{method}_2.csv")
    myP.p_df_3.to_csv(f"{method}_3.csv")

    return myP.p_df_2

lgbm_ = trading(lgbm, "lgbm")
exit()
arima_ = trading(arima, "arima")
comb_ = trading(comb, "comb")

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
plt.savefig("Cumulative_Trading.jpg")


# correlation
############################################################################################################

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
    """ Calculate Expected Shortfall (ES) using KDE. """
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
