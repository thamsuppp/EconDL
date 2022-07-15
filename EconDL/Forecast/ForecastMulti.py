import pandas as pd
import numpy as np
import random 
from datetime import datetime

from EconDL.utils import invert_scaling
from EconDL.predict_nn import predict_nn_new, predict_nn_old, predict_ml_model

# Repeat-level class to generate multi-period forecasts
class ForecastMulti:

  def __init__(self, run_name, Y_train, Y_test, multi_forecasting_params, device = None):

    self.run_name = run_name
    self.folder_path = f'../results/{self.run_name}'

    self.Y_train = Y_train
    self.Y_test = Y_test

    self.Y_all = np.concatenate([Y_train, Y_test], axis = 0)

    self.h = multi_forecasting_params['forecast_horizons']
    self.test_size = multi_forecasting_params['test_size'] #T
    self.reestimation_window = multi_forecasting_params['reestimation_window']
    self.R = int(self.test_size / self.reestimation_window)
    self.num_repeats = multi_forecasting_params['num_repeats']
    self.num_inner_bootstraps = multi_forecasting_params['num_inner_bootstraps']
    self.num_sim_bootstraps =  multi_forecasting_params['num_sim_bootstraps'] # B

    self.n_lag_linear = multi_forecasting_params['n_lag_linear']
    self.n_lag_d = multi_forecasting_params['n_lag_d']
    self.n_var = multi_forecasting_params['n_var']
    self.var_names = multi_forecasting_params['var_names']

    self.forecast_method = multi_forecasting_params['forecast_method']

    self.device = device

  # @title Bootstrap Forecasting Function (Joint Estimation Forecasting Method)
  # Shape of mat_x_d_all: first n_lag_linear x self.n_var, then n_lag_d x self.n_var (hence 3 + 24 = 27)

  # Base code: Make 0-h self.horizon predictions for 1 bootstrap, for 1 tiemstep
  # New for 14 Dec: input the error_ids so we can keep constant across different models
  def predict_one_bootstrap_new(self, newx, results, nn_hyps):
    
    Y_train = results['Y_train']
    n_lag_linear = nn_hyps['n_lag_linear']
    n_lag_d = nn_hyps['n_lag_d']
    # Input dim of newx: 1 dimensional

    # Store the average OOB predictions across all inner bootstraps
    oob_preds = results['pred_in']
    # oob_res: set of OOB error vectors to sample from for the iterated forecasts
    oob_res = Y_train - results['pred_in']

    # Remove all NA values so that we don't sample NAs
    a = pd.DataFrame(oob_res)
    oob_res = np.array(a.dropna())

    # Create array to store the predictions for each bootstrap
    fcast = np.zeros((self.h+1, self.n_var))
    fcast[:] = np.nan

    fcast_cov_mat = np.zeros((self.h+1, self.n_var, self.n_var))
    fcast_cov_mat[:] = np.nan

    # Create vectors to store the 3 segments of the x input: linear, nonlinear and time
    new_in_linear = np.zeros(n_lag_linear * self.n_var)
    new_in_nonlinear = np.zeros(n_lag_d * self.n_var)
    new_in_time = np.zeros(newx.shape[0] - (n_lag_linear + n_lag_d) * self.n_var)

    # Start with the initial input
    new_in_linear = newx[0:(n_lag_linear * self.n_var)]
    new_in_nonlinear = newx[(n_lag_linear * self.n_var):((n_lag_linear + n_lag_d) * self.n_var)]
    new_in_time = newx[((n_lag_linear + n_lag_d) * self.n_var):]

    for period in range(1, self.h + 1):

      if period != 1: # Excluding first period (i.e. there are forecasts)
        # Update simulated prediction for previous period to the current period's newx data
        if n_lag_linear == 1:
          new_in_linear = fcast[period-1, :]
        else:
          new_in_linear = np.hstack([fcast[period-1, :], new_in_linear[:(len(new_in_linear) - self.n_var)]])

        new_in_nonlinear = np.hstack([fcast[period-1, :], new_in_nonlinear[:(len(new_in_nonlinear) - self.n_var)]])

      # Conduct MARX transformation on the nonlinear layer
      new_data_marx = new_in_nonlinear.copy()
      for lag in range(2, n_lag_d + 1):
        for var in range(self.n_var):
          who_to_avg = list(range(var, self.n_var * (lag - 1) + var + 1, self.n_var))
          new_data_marx[who_to_avg[-1]] = new_in_nonlinear[who_to_avg].mean()
      
      # Combine the first n_lag_linear lags, with the MARX data, to get the full input vector
      new_data_all = np.hstack([new_in_linear, new_data_marx, new_in_time])
      new_data_all = np.expand_dims(new_data_all, axis = 0)

      # Now we self.have new_data_all
      
      # Use estimated model to make prediction with the generated input vector
      pred, cov = predict_nn_new(results, new_data_all, self.device)

      # Cholesky the cov mat to get C matrix
      cov = np.squeeze(cov, axis = 0)
      fcast_cov_mat[period, :, :] = cov
      c_t = np.linalg.cholesky(cov)
      
      if period != self.h:
        # Sample 1 shock from normal distribution
        sim_shock = np.random.multivariate_normal([0] * self.n_var, np.eye(self.n_var), size = 1)

        # Convert the shock back into residual, add this to the series
        sim_resid = np.matmul(sim_shock, c_t.T)
        
        fcast[period, :] = pred + sim_resid
      else: # if last period (h) - then no need to add any sampled errors
        fcast[period, :] = pred
      
    return fcast 

  # @title Bootstrap Forecasting Function (Old Method - works for ML models)

  # Shape of mat_x_d_all: first n_lag_linear x self.n_var, then n_lag_d x self.n_var (hence 3 + 24 = 27)
  # Base code: Make 0-h self.horizon predictions for 1 bootstrap, for 1 tiemstep
  # New for 14 Dec: input the error_ids so we can keep constant across different models
  def predict_one_bootstrap_old(self, newx, results, nn_hyps, model = 'VARNN'):

    Y_train = results['Y_train']
    n_lag_linear = nn_hyps['n_lag_linear']
    n_lag_d = nn_hyps['n_lag_d']
    # Input dim of newx: 1 dimensional

    # Store the average OOB predictions across all inner bootstraps
    oob_preds = results['pred_in']
    # oob_res: set of OOB error vectors to sample from for the iterated forecasts
    oob_res = Y_train - results['pred_in']

    # Remove all NA values so that we don't sample NAs
    a = pd.DataFrame(oob_res)
    oob_res = np.array(a.dropna())
    #print('oob_res', oob_res)

    # Create array to store the predictions for each bootstrap
    fcast = np.zeros((self.h +1, self.n_var))
    fcast[:] = np.nan

    # Create vectors to store the 3 segments of the x input: linear, nonlinear and time
    new_in_linear = np.zeros(n_lag_linear * self.n_var)
    new_in_nonlinear = np.zeros(n_lag_d * self.n_var)
    new_in_time = np.zeros(newx.shape[0] - (n_lag_linear + n_lag_d) * self.n_var)

    # Start with the initial input
    new_in_linear = newx[0:(n_lag_linear * self.n_var)]
    new_in_nonlinear = newx[(n_lag_linear * self.n_var):((n_lag_linear + n_lag_d) * self.n_var)]
    new_in_time = newx[((n_lag_linear + n_lag_d) * self.n_var):]

    for period in range(1, self.h + 1):

      if period != 1: # Excluding first period (i.e. there are forecasts)
        # Update simulated prediction for previous period to the current period's newx data
        if n_lag_linear == 1:
          new_in_linear = fcast[period-1, :]
        else:
          new_in_linear = np.hstack([fcast[period-1, :], new_in_linear[:(len(new_in_linear) - self.n_var)]])

        new_in_nonlinear = np.hstack([fcast[period-1, :], new_in_nonlinear[:(len(new_in_nonlinear) - self.n_var)]])

      # Conduct MARX transformation on the nonlinear layer
      new_data_marx = new_in_nonlinear.copy()
      for lag in range(2, n_lag_d + 1):
        for var in range(self.n_var):
          who_to_avg = list(range(var, self.n_var * (lag - 1) + var + 1, self.n_var))
          new_data_marx[who_to_avg[-1]] = new_in_nonlinear[who_to_avg].mean()
      
      # Combine the first n_lag_linear lags, with the MARX data, to get the full input vector
      new_data_all = np.hstack([new_in_linear, new_data_marx, new_in_time])
      new_data_all = np.expand_dims(new_data_all, axis = 0)
      
      if model in ['RF', 'XGBoost']:
        pred = predict_ml_model(results, new_data_all)
      else: # VARNN
      # Use estimated model to make prediction with the generated input vector
        pred = predict_nn_old(results, new_data_all, self.device)

      # Add the sampled error if not the last period
      if period != self.h:
        # Randomly sample with replacement an error vector from the OOB distribution
        sample_id = random.choice(range(oob_res.shape[0]))
        #sample_id = error_ids[period] # this cannot work anymore because we could be sampling NAs
        
        # Get the error vector
        sampled_error = oob_res[sample_id, :]
        fcast[period, :] = pred + sampled_error
      else: # if last period (h) - then no need to add any sampled errors
        fcast[period, :] = pred
      
    return fcast 

  

  def conduct_multi_forecasting_wrapper(self, X_train, X_test, results, nn_hyps):
    # Fix the shock ids across the different models
    #error_ids = np.array(random.choices(range(X_train.shape[0]), k = self.h * self.num_sim_bootstraps * self.test_size))
    #error_ids = error_ids.reshape((self.h, self.num_sim_bootstraps, self.test_size))

    if self.forecast_method == 'new':
        predict_fn = self.predict_one_bootstrap_new
    else:
      predict_fn = self.predict_one_bootstrap_old

    results['Y_train'] = self.Y_train
    results['Y_test'] = self.Y_test

    FCAST = np.zeros((self.h + 1, self.n_var, self.num_sim_bootstraps, self.test_size, self.R))
    FCAST[:] = np.nan
    r = 0
    # For every timestep (in the future), for every bootstrap, get the self.h-th self.horizon prediction
    for t in range(r * self.reestimation_window, self.test_size):
      if t % 5 == 0:
        print(f'Time {t}, {datetime.now()}')
      for b in range(self.num_sim_bootstraps):
        if t == r * self.reestimation_window: # first sample (X_train is continually changed so can take last value of X_train)
          FCAST[:, :, b, t, r] = predict_fn(X_train[-1, :], results, nn_hyps)
        else: # not the first sample (X_test is unchanged so can just index t)
          FCAST[:, :, b, t, r] = predict_fn(X_test[t, :], results, nn_hyps)
    
    return FCAST


