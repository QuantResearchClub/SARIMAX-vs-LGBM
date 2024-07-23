import optuna
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import lightgbm as lgb
import numpy as np

# ARIMA
# -----------------------------------------------
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

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
    # DAWEI
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
    
    
    
    # ARIMA - CUIYUAN
    # --------------------------------------------------------------------------------------------
    # The (p,d,q) order of the model for the autoregressive, differences, and moving average components. 
    # d is always an integer, while p and q may either be integers or lists of integers.
    def ARIMA_predict(self, target, column, p, d, q):
        testSet = self.testSet
        trainSet = self.validSet
        print(trainSet)
        print(testSet)

        model = sm.tsa.statespace.SARIMAX(
            endog = trainSet[target], 
            exog  = trainSet[column],
            order = (p,d,q),
            seasonal_order = (0, 1, 1, 12)
        )
        arima_result = model.fit()
        print(arima_result.summary())
        test_pred = arima_result.forecast(
            steps = testSet.shape[0],
            exog  = testSet[column]
        )
        # mse = mean_squared_error(test_pred, testSet)
        test_pred.index = testSet.index
        print(test_pred)
        print("##########################################################")
        print(testSet[target])

        plt.plot(testSet[target][1:], color = "blue")
        plt.plot(test_pred[1:], color = "green")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.show()

    # --------------------------------------------------------------------------------------------