from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import numpy as np
import os


class ForecastBenchmarks:

  def __init__(self, dataset, multi_forecasting_params, run_name):
    
    self.dataset = dataset
    self.Y_train = None
    self.Y_test = None

    self.run_name = run_name
    self.folder_path = f'../results/{run_name}'
    # Create benchmark folder if not exist yet
    self.benchmark_folder_path = f'{self.folder_path}/benchmarks'
    if os.path.isdir(self.benchmark_folder_path) == False:
      os.mkdir(self.benchmark_folder_path)

    self.h = multi_forecasting_params['forecast_horizons']
    self.test_size = multi_forecasting_params['test_size'] #T
    self.reestimation_window = multi_forecasting_params['reestimation_window']
    self.R = int(self.test_size / self.reestimation_window)
    self.num_repeats = multi_forecasting_params['num_repeats']

    self.n_lag_linear = multi_forecasting_params['n_lag_linear']
    self.n_lag_d = multi_forecasting_params['n_lag_d']
    self.n_var = multi_forecasting_params['n_var']
    self.var_names = multi_forecasting_params['var_names']

    self._process_dataset()

  def _process_dataset(self):
    
    Y_all = self.dataset[self.var_names]
    # Remove the first n_lag_d observations
    Y_all = Y_all.iloc[self.n_lag_d:, :].reset_index(drop=True)
    n_obs = Y_all.shape[0]
    train_split_id = n_obs - self.test_size

    Y_train = np.array(Y_all.iloc[:train_split_id, :])
    Y_test = np.array(Y_all.iloc[train_split_id:, :])

    self.Y_train = Y_train
    self.Y_test = Y_test
    self.Y_all = np.array(Y_all)


  def conduct_multi_forecasting_benchmarks(self):
    self.expanding_window_VAR()
    self.rolling_window_VAR()
    self.expanding_window_AR()
    self.rolling_window_AR()

  ### Expanding Window VAR
  def expanding_window_VAR(self):
    # Test prediction array
    FCAST = np.zeros((self.h+1, self.n_var, self.test_size, self.R))
    FCAST[:] = np.nan

    # For every time through the re-estimation window
    for r in range(self.R):
      print(f'Re-estimation window {r}, {datetime.now()}')
      # Training data is the data available till that moment
      Y_train = self.Y_all[:-(self.test_size - self.reestimation_window * r), :]
      
      # Estimate the VAR model with available data
      var_model = VAR(Y_train)
      results = var_model.fit(self.n_lag_linear)
      for t in range(self.reestimation_window * r, self.test_size):
        if t == self.reestimation_window * r:
          FCAST[1:, :, t, r] = results.forecast(Y_train[-self.n_lag_linear:, :], steps = self.h)
        # In between case when you need to combine Y_train and some bit of Y_test
        elif t < self.reestimation_window * r + self.n_lag_linear:
          train_obs_needed = self.n_lag_linear - (t - self.reestimation_window * r)
          test_obs_needed = t - self.reestimation_window * r
          Y_in = np.concatenate([Y_train[-train_obs_needed:, :], self.Y_test[:test_obs_needed, :]], axis = 0)
          FCAST[1:, :, t, r] = results.forecast(Y_in, steps = self.h)
        else:
          FCAST[1:, :, t, r] = results.forecast(self.Y_test[(t-self.n_lag_linear):t, :], steps = self.h)

    with open(f'{self.folder_path}/expanding_var_forecasting.npz', 'wb') as f:
        np.save(f, FCAST)

  def rolling_window_VAR(self):
    # Test prediction array
    FCAST = np.zeros((self.h +1, self.n_var, self.test_size, self.R))
    FCAST[:] = np.nan

    window_length = 40
    # For every time through the re-estimation window
    for r in range(self.R):
      print(f'Re-estimation window {r}, {datetime.now()}')
      # Training data is the data available till that moment - *restricted to the length of window
      Y_train = self.Y_all[-(self.test_size - self.reestimation_window * r + window_length):-(self.test_size - self.reestimation_window * r), :]
      # Estimate the VAR model with available data
      var_model = VAR(Y_train)
      results = var_model.fit(self.n_lag_linear)
      for t in range(self.reestimation_window * r, self.test_size):
        if t == self.reestimation_window * r:
          FCAST[1:, :, t, r] = results.forecast(Y_train[-self.n_lag_linear:, :], steps = self.h)
        # In between case when you need to combine Y_train and some bit of Y_test
        elif t < self.reestimation_window * r + self.n_lag_linear:
          train_obs_needed = self.n_lag_linear - (t - self.reestimation_window * r)
          test_obs_needed = t - self.reestimation_window * r
          Y_in = np.concatenate([Y_train[-train_obs_needed:, :], self.Y_test[:test_obs_needed, :]], axis = 0)
          FCAST[1:, :, t, r] = results.forecast(Y_in, steps = self.h)
        else:
          FCAST[1:, :, t, r] = results.forecast(self.Y_test[(t-self.n_lag_linear):t, :], steps = self.h)

    with open(f'{self.folder_path}/rolling_var_forecasting.npz', 'wb') as f:
        np.save(f, FCAST)
  
  def get_ar_forecasts(self, y_in, results_coefs, h):
    # y_in should be of length 'ar_lags' - all you need to start iterating forward
    
    # Store both train and test 
    y_all = np.zeros((y_in.shape[0] + h))
    y_all[:] = np.nan
    y_all[:y_in.shape[0]] = y_in

    for self.horizon in range(1, self.h+1):
      # Get the input lags for 
      y_in_this = y_all[(y_in.shape[0] -1 -4 + self.horizon):(y_in.shape[0] -1 + self.horizon)]
      # Evaluate the AR equation to get the prediction
      y_all[y_in.shape[0] - 1 + self.horizon] = results_coefs[0] + np.dot(results_coefs[1:], y_in)

    # Return self.h predictions
    return y_all[y_in.shape[0]:]

  def expanding_window_AR(self):
    ### Expanding Window AR(4)
    self.reestimation_window = 1
    R = int(self.test_size / self.reestimation_window)

    ar_lags = 4

    FCAST = np.zeros((self.h+1, self.n_var, self.test_size, R))
    FCAST[:] = np.nan

    # For every time through the re-estimation window
    for r in range(R):
      print(f'Re-estimation window {r}, {datetime.now()}')
      # Training data is the data available till that moment
      Y_train = self.Y_all[:-(self.test_size - self.reestimation_window * r), :]

      if Y_train.shape[0] != 0:

        for var in range(self.n_var):
          y_train = Y_train[:, var]
          y_test = self.Y_test[:, var]
          arima_model = ARIMA(y_train, order = (ar_lags,0,0))
          results = arima_model.fit()
          results_coefs = results.params[0:(ar_lags + 1)]
        
          for t in range(self.reestimation_window * r, self.test_size):
            # Different cases of getting the input y_in
            if t == self.reestimation_window * r:
              y_in = y_train[-ar_lags:]
            # In between case when you need to combine Y_train and some bit of Y_test
            elif t < self.reestimation_window * r + ar_lags:
              train_obs_needed = ar_lags - (t - self.reestimation_window * r)
              test_obs_needed = t - self.reestimation_window * r
              y_in = np.concatenate([y_train[-train_obs_needed:], y_test[:test_obs_needed]], axis = 0)
            else:
              y_in = y_test[(t-ar_lags):t]

            FCAST[1:, var, t, r] = self.get_ar_forecasts(y_in, results_coefs, self.h)

    with open(f'{self.folder_path}/expanding_ar_forecasting.npz', 'wb') as f:
        np.save(f, FCAST)

  ### Rolling Window AR(4)
  def rolling_window_AR(self):
  
    self.reestimation_window = 1
    R = int(self.test_size / self.reestimation_window)

    ar_lags = 4

    FCAST = np.zeros((self.h+1, self.n_var, self.test_size, R))
    FCAST[:] = np.nan

    window_length = 40
    # For every time through the re-estimation window
    for r in range(R):
      print(f'Re-estimation window {r}, {datetime.now()}')
      # Training data is the data available till that moment
      Y_train = self.Y_all[-(self.test_size - self.reestimation_window * r + window_length):-(self.test_size - self.reestimation_window * r), :]

      if Y_train.shape[0] != 0:

        for var in range(self.n_var):
          y_train = Y_train[:, var]
          y_test = self.Y_test[:, var]
          arima_model = ARIMA(y_train, order = (ar_lags,0,0))
          results = arima_model.fit()
          results_coefs = results.params[0:(ar_lags + 1)]
        
          for t in range(self.reestimation_window * r, self.test_size):
            # Different cases of getting the input y_in
            if t == self.reestimation_window * r:
              y_in = y_train[-ar_lags:]
            # In between case when you need to combine Y_train and some bit of Y_test
            elif t < self.reestimation_window * r + ar_lags:
              train_obs_needed = ar_lags - (t - self.reestimation_window * r)
              test_obs_needed = t - self.reestimation_window * r
              y_in = np.concatenate([y_train[-train_obs_needed:], y_test[:test_obs_needed]], axis = 0)
            else:
              y_in = y_test[(t-ar_lags):t]

            FCAST[1:, var, t, r] = self.get_ar_forecasts(y_in, results_coefs, self.h)

    with open(f'{self.folder_path}/rolling_ar_forecasting.npz', 'wb') as f:
        np.save(f, FCAST)