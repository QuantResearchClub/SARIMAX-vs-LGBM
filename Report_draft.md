![image](https://github.com/user-attachments/assets/2f1938cb-7ab2-40cb-84ed-26163bb79f2a)
Group 5:
# Abstract
### 1.1. Our group target:
The topic of our group to investigate on the performance of predictions among different time-series models, we find there are not adequate research on this topic, hence introduces a comparison between them.

hence find a prediction model of the daily future log-return among shares in Nasdaq, through the positive and negative information of the predicted result, could assist judging the rightness of portofolio selections. In specific, these predictions are made by algorithms include ARIMA, LightGBM decision trees and MOV, we made comaprisions among these algorithms and hence provide suggestions of using them.

### 1.2. Our group suggestion:
We suggested by using a composition model of ARIMA and LGBM for limited-time period training of the shares predictions, for reasons we find: ...


### 1.3. Our group roles: (Draft, need edited)
#### Song Haoning
- Provide documents of the detailed indicators.
- Provide backtests among the models and optimize the parameters we used.
- Provide basic analyzation of results. 

#### Zhang Dawei:
- Provide the basic scheme of the whole project.
- Provide the script for LGBM part
- Provide the script for MOV part.
- Provide the data displaying part.

#### Liu Cuiyuan:
- Integrate and test the code developed for the project.
- Provide the data procession part scripts, include downloading, sorting, cleaning and training.
- Provide the ARIMA part for predictions.

# Outcomes
## 2. Introductions:
While Kobiela (2022) suggests that ARIMA model performed (多少程度) better than LSTM model did on NASDAQ stock exchange data by providing a better accuracy as well as stability.

## 3. Data:
Our dataset is built on the Nasdaq-100 index, we picked the historical constitutents of this data in 2024 from Jan 1st to June 30th for training, and introduces the data in July for backtests and further testings. The reason we choose Nasdaq 100 is because it is consist of the 100 biggest non-financial companies (Nasdaq, 2024), which are of good investiment diversification, high liquidity and stable market structure, hence fits the quant trading. (Hasbrouck, 2007). 

We downloaded the data from the yahoo finance and the original data consists of the daily data of close price, open price, high price, low price and the trading volume of every ticker listed. For a further procession of these sources, we introduce further indicators includes

[1] https://indexes.nasdaq.com/docs/Methodology_NDX.pdf
[2] Hasbrouck, J. (2007). Empirical Market Microstructure: The Institutions, Economics, and Econometrics of Securities Trading. Oxford University Press.
## 4. Model:
### 4.1 MOV Model
We calculate expected return and covariance matrix from the training dataset and optimize the portfolio weight to maximize return for a given level of risk

### 4.2 LGBM Model
LightGBM is a gradient boosting ensemble method based on decision trees and could be used in classification and regression. LightGBM models are using the features extracted from the training data. The advantage of LGBM is faster speed and high accuracy with lower memory usage
- https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/
### 4.3 ARIMA Model
The ARIMA model is a stationary time series model to predict each individual stock respectively. As most market data show trends, the purpose of differencing is to remove any trends or seasonal structures.
- https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp 

### 4.4 Ensemble Model
Then we use the ensemble method to average scaled predictions of ARIMA and LightGBM. As stable and could share index follows the time series model. We use the scale predictions from each model to the same range then calculate the average of the scaled predictions to form the ensemble prediction.

## 5. Performance Evaluation:
### 5.1 Portofolio Picked
We calculate the log return as the y-axis of our model as logarithm reduced the calculation cost and we predict the log retun of each stock at time t-1, then select the top 5 stocks with the highest predicted return to be our buying set. 

### 5.2 Cumulative Return
TICKERSLST_USE1 = ['AAPL','MSFT','NVDA','GOOG','AMZN','META','TSM','LLY','TSLA','AVGO']
![alt text](image-1.png)
The benchmark is the cumulative return without applying any specific strategy andnd it shows a decrease, reaching a cumulative return of about -0.06 by the end of the month. LGBM's performance is similar to the benchmark but slightly higher throughout the period, ending with a cumulative return of approximately -0.05. ARIMA is very similar with the benchmark and by the end of the month the cumulative return is around -0.065 which slightly lower than the benchmark. MVO shows shows worst performance at the end of the month which is extremely lower than the benchmark. Ensamble shows the portfolio strategy which also perform decrease feature in cumulative returns and the cumulative return at the end of the month is around -0.12.

TICKERSLST_USE2 = ['WBD', 'ODFL', 'CMCSA', 'PYPL', 'KHC', 'GILD', 'GEHC', 'FAST', 'BKR', 'ADP']
![alt text](image.png)
The benchmark is the cumulative return without applying any specific strategy andnd it shows a steady increase, reaching a cumulative return of about 0.1 by the end of the month. LGBM's performance is similar to the benchmark but slightly higher throughout the period, ending with a cumulative return of approximately 0.12. ARIMA also perform a similarly trend with benchmark and by the end of the month the cumulative return is around 0.12 which slightly higher than the benchmark. MVO shows a brief increase initially but then drops sharply and shows worst performance at the end of the month. Ensamble shows the portfolio strategy which also perform increase feature in cumulative returns and the purple line is the most close line with benchmark.

### 5.3 QQ-plot and data analysis
![alt text](image-2.png)
All of the QQ plots show heavy tail round the range from -2.0 to -0.55 which means that there exist some bias in the range of extreme value. Most dots of LGBM followed the 45-degree reference line that the residuals or returns are approximately normally distributed. Very less dots followed the diagonal of MOV. Points closely follow the 45-degree reference line, indicating that the residuals or returns from the ARIMA model are newar normally distributed. For ensemble methods, points generally follow the 45-degree reference line with minor deviations. The real return shows that points deviate from the diagonal, especially at the tails.

### 5.4 Expected Shortfall and Value at risk
![alt text](image-3.png)
This table shows the value at risk and expected shortfall of all models in difference confidence interval. At the 90% confidence interval MOV has the largest VaR which is -5.97% and it indicates MOV may have the highest potential loss. LGBM has the lowest value at risk in 90% confidence interval. At the 95% confidence interval, MOV still has the highest VaR and ARIMA has the lowest VaR which is -3.93%.

At the 90% confidence level, the MVO strategy has the highest ES at -6.02%, indicating the highest average loss beyond VaR. The ARIMA strategy has the lowest ES at -4.03%. At the 95% confidence level, the LGBM strategy has the highest ES at -6.08%. The ARIMA strategy again has the lowest ES at -4.13%.

The MVO strategy exhibits the highest potential and expected losses (VaR and ES), indicating higher risk in extreme market conditions.

The ARIMA strategy consistently shows the lowest potential and expected losses, indicating lower risk.
## 6. Conclusions


# Reference
[1] https://indexes.nasdaq.com/docs/Methodology_NDX.pdf
  
