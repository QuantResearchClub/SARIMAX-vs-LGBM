## Test 01: test the k-folds split of the dataset.
from Info import *
from Functions import *
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

def test01():
    currShare = Share()
    currShare.setupShare(TICKERSLST[0], "Close", True, "")
    df_split = currShare.split_set

    n_fold = 10
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

# Test 02 : Defaulted Inputs:
def test02():
    def nfn(a = 3):
        print(a)
    nfn(7)
    nfn()

# Test 03 : test portofolio part
def test03():
    df_a = pd.read_csv("DataResult\\Result_ARIMA.csv")
    myP = Portofolio()
    myP.set_up(df_a, "arima")
    myP.trading(myP.data, "predicted_Y1", 5)
    print(myP.data)
    print(myP.p_df_1)
    print(myP.p_df_2)
    print(myP.p_df_3)

# test03()

# Test 04: test mvo
def test04():
    pass