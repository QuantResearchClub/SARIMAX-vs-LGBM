
## LightGBM的优势和劣势
LightGBM的训练器是相对于XGBosst的概念, 它最显著的优点在于将重复的数据进行组合再一起得到一个平均值(分类树状图), 这样降低了统计上的成本。


## Kfolds 交叉训练
因为数据量不够大, 我认为如果效果不理想, 可以采取kfolds的方式进行交叉验证以优化训练, 这里提供了第一个参数k

KFold divides all the samples in groups of samples, called folds (if $k = n$, this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using $k - 1$ folds, and the fold left out is used for test.

我们采用的是`TimeSeriesSplit`, 问题是为什么采取3-folds进行训练以及对应集合的分开是如何影响结果的?
```python
## Test 01: test the k-folds split of the dataset.
from Info import *
from Functions import *
from sklearn.model_selection import TimeSeriesSplit

currShare = Share()
currShare.setupShare(TICKERSLST[0], "Close", True, "")
df_split = currShare.split_set

n_fold = 3
folds = TimeSeriesSplit(
            n_splits = n_fold,
            gap = 0,                  # gap between the train and test side of the splits. 
            max_train_size = 10000,
            test_size = 10,           # about 10% of training data
        )
splits = folds.split(df_split['train_x'], df_split['train_y'])
for fold_n, (train_index, valid_index) in enumerate(splits):
    print(fold_n, train_index, valid_index)
    print(df_split['train_x'].iloc[train_index])
    print(df_split['train_x'].iloc[valid_index])
```


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_iris

# 加载数据集（以鸢尾花数据集为例）
iris = load_iris()
X = iris.data
y = iris.target

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100)

# 创建十折交叉验证对象
kfold = KFold(n_splits=10)

# 执行十折交叉验证
scores = cross_val_score(rf_classifier, X, y, cv=kfold)

# 输出每折的准确率
for i, score in enumerate(scores):
    print("Fold {}: {:.4f}".format(i+1, score))

# 输出平均准确率
print("Average Accuracy: {:.4f}".format(scores.mean()))
```


策略:
logret > 0 则采取买进



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing, metrics
from datetime import date
from scipy.stats import ttest_rel

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
```



## 数据对比
下面是简单的数据对比, 
```python
```