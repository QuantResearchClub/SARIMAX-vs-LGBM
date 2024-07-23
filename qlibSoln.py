"""
from gluonts.ext.r_forecast import RForecastPredictor
# build model
arima_estimator = RForecastPredictor(freq='1D', prediction_length = args.horizon, method_name="arima")
# Predicting 
forecast_df = pd.DataFrame(columns=['id', 'target_start_date', 'point_fcst_value'])  # df_pred
for entry_, forecast_ in tqdm(zip(training_data, estimator.predict(training_data))):
     id = entry_["id"]
     forecast_df = forecast_df.append( 
                       pd.DataFrame({"id": id,
                                     "target_start_date": forecast_.index.map(lambda s: s.strftime('%Y%m%d')),
                                     "point_fcst_value": forecast_.median}))



# build model
arima_estimator = RForecastPredictor(freq='1D', prediction_length=args.horizon, method_name="arima")
# Predicting 
forecast_df = pd.DataFrame(columns=['id', 'target_start_date', 'point_fcst_value'])  # df_pred
for entry_, forecast_ in tqdm(zip(training_data, estimator.predict(training_data))):
     id = entry_["id"]
     forecast_df = forecast_df.append( 
                       pd.DataFrame({"id": id,
                                     "target_start_date": forecast_.index.map(lambda s: s.strftime('%Y%m%d')),
                                     "point_fcst_value": forecast_.median}))
"""