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
## 4. Methodology:
### 4.1 MOV Model

### 4.2 LGBM Model

### 4.3 ARIMA Model

### 4.4 Ensemble Model

## 5. Performance Evaluation:
### 5.1 Portofolio Picked

### 5.2 Cumulative Return

### 5.3 QQ-plot and data analysis

### 5.4 Expected Shortfall and Value at risk

## 6. Conclusions


# Reference

  
