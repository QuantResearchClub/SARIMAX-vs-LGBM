"""
    # Training Set
    s_timestamp = '2024-1-1'
    f_timestamp = '2024-6-30'
    yf_downloadDataClass(s_timestamp, f_timestamp, 2)

    # Validation Set
    s_timestamp = '2024-7-1'
    f_timestamp = '2024-7-31'
    yf_downloadDataClass(s_timestamp, f_timestamp, 2)
"""

from runFunctions import *
from timeSeries import *

ts_ts = '2023-1-1' 
tf_ts = '2023-12-31'
vs_ts = '2024-1-1'
vf_ts = '2024-6-30'
es_ts = '2024-7-1'
ef_ts = '2024-7-31'

def downloadData():
    myKernel = Kernel()
    myKernel.yf_downloadData(ts_ts, tf_ts, 2, "TrainingSet")
    myKernel.yf_downloadData(vs_ts, vf_ts, 2, "ValidationSet")
    myKernel.yf_downloadData(es_ts, ef_ts, 2, "TestingSet")

def runData(targetShare: str, loc: str, ts_ts, tf_ts):
    t_DS = pd.read_csv(loc + "_".join([ts_ts, tf_ts, targetShare]) + ".csv")
    my_tShare = TradeData()
    my_tShare.setup(t_DS, ts_ts, tf_ts, targetShare)
    my_tShare.std()
    my_tShare.bollinger_strat(14, 2)
    my_tShare.heikin_ashi()
    my_tShare.logret()
    my_tShare.data = my_tShare.data.fillna(0)
    my_tShare.to_datetime("Date")
    
    my_tShare.data.to_csv(loc + "_".join([ts_ts, tf_ts, targetShare, "Processed.csv"]))
    return my_tShare

def run_share(share: str):
    temp_result_lst = ["TrainingSet", "ValidationSet", "TestingSet"]
    temp_result_dict = dict(zip(temp_result_lst, [] * len(temp_result_lst)))

    for verb in temp_result_lst:
        s_ts, f_ts =  ts_ts, tf_ts
        if (verb == "ValidationSet"): s_ts, f_ts =  vs_ts, vf_ts
        if (verb == "TestingSet"): s_ts, f_ts =  es_ts, ef_ts
        temp_result = runData(share, "Dataset\\" + share + "\\" + verb + "\\", s_ts, f_ts)
        ans = temp_result.data.copy()
        temp_result_dict[verb] = ans

    myTStrain = TimeSeries()
    myTStrain.setup_TStrain(
        df_predict = pd.DataFrame(),
        df_train = temp_result_dict["TrainingSet"],
        df_test  = temp_result_dict["TestingSet"],
        df_valid = temp_result_dict["ValidationSet"],
        target = "return",
        ipt_cols = ['High','Low','Open','Volume', "Close", 'sd_m', 'sd_w', 'boll_high', 'boll_low', 'open_HA', 'close_HA'] 
    )

    myTStrain.ARIMA_predict(myTStrain.target, myTStrain.columns, 1, 1, 1)


"""
    myTStrain.optuna_paras(n_trials = 10)
    print(myTStrain.df_predicted)
"""
# downloadData()
run_share(share = "XEL")

