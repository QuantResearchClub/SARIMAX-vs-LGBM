
拿到收盘价单
- Get to know what kind of data can we access from the yfinance? Open, High, Low, Close, Volume
- Get to know how to store these data? pdDataframe
- Get to know what data we need for QLib?
- Write good maniputable functions for processing the Qlib data. Done

Qlib
- 优点 好像非常便捷
- 缺点 在安装环境上遇到了比较大的困难
- 缺点 在函数上无法使用, 非常难受 (包括qlib.contribut以及其它函数无法使用)

改用gluonts
from gluonts.ext.r_forecast import RForecastPredictor
https://ts.gluon.ai/stable/api/gluonts/gluonts.ext.r_forecast.html
https://zhihu.com/column/c_1258782761349410816

![alt text](image.png)


旧版本 不知道该怎么改:
```python
import optuna
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import lightgbm as lgb
import numpy as np

class TimeSeries():
    def __init__(self):
        self.n_fold = 5
        self.df_predicted = pd.DataFrame()
        self.trainSet = pd.DataFrame()
        self.validSet = pd.DataFrame()
        self.testSet = pd.DataFrame()
        self.target = ""
        self.columns = []
    
    # Time Series Training set-up
    # --------------------------------------------------------------------------------------------
    def setup_TStrain(self, df_predict, df_train, df_valid, df_test, target, ipt_cols):
        self.df_predicted = df_predict
        self.trainSet = df_train
        self.validSet = df_valid
        self.testSet = df_test
        self.target = target
        self.columns = ipt_cols
        return self

    # Objective Function Definition
    # --------------------------------------------------------------------------------------------
    def objective_fn(self, trial):
        params = {# 'num_leaves': trial.suggest_int('num_leaves', 2, 50),
                    'num_iterations': trial.suggest_int('num_iterations', 300, 500),  
                    'max_depth': trial.suggest_int('max_depth', 14, 18),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.01),
                    "boosting_type": "gbdt",
                    "metric": 'binary_logloss',
                    "verbosity": -1,
                    'random_state': 13,
                }
        
        y_preds = np.zeros(self.validSet.shape[0])
        y_oof   = np.zeros(self.trainSet.shape[0])
        
        for fold_n, (train_index, valid_index) in enumerate(self.trainSet.shape[0], self.validSet.shape[0]):
            X_train = self.trainSet[self.columns].iloc[train_index]
            X_valid = self.validSet[self.columns].iloc[train_index]
            y_train = self.trainSet[self.target].iloc[train_index]
            y_valid = self.validSet[self.target].iloc[valid_index]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid)
            clf = lgb.train(params, dtrain, 50, valid_sets = [dtrain, dvalid])

            y_pred_valid = clf.predict(X_valid, num_iteration = clf.best_iteration)
            y_oof[valid_index] = y_pred_valid
            y_preds += clf.predict(self.validSet['test'][self.columns], num_iteration=clf.best_iteration) / self.n_fold
        
        self.testSet['predicted'] = np.where(y_preds >= 0.4, 1, 0)
        self.df_predicted = self.testSet[['Date','return', self.target, 'predicted']]

        # binary: 1=hold, 0=do not hold. Include estimated transaction cost
        self.df_predicted['strat'] = (
            self.df_predicted['return'] * self.df_predicted['predicted'].shift(1) - 0.5 * ((
                self.df_predicted['predicted'] != self.df_predicted['predicted'].shift(1)
            ).astype(int)/ self.testSet.iloc[-1]['Open'])
        )
        return self.df_predicted['strat'].sum()
    
    # Optuna training
    # --------------------------------------------------------------------------------------------
    def optuna_paras(self, n_trials):
        # Create an Optuna study object
        study = optuna.create_study(direction='maximize')

        # Optimize the study
        study.optimize(self.objective_fn, n_trials = n_trials)

        # Output the best hyperparameters and the best score
        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")
```

可恶啊

