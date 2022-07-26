import numpy as np
import matplotlib.pyplot as plt
import os

class ForecastMultiEvaluation:

  def __init__(self, run_name, multi_forecasting_params, Y_train, Y_test):
    
    self.run_name = run_name
    self.folder_path = f'results/{self.run_name}'
    self.image_folder_path = f'{self.folder_path}/images'

    self.Y_train = Y_train
    self.Y_test = Y_test
    self.Y_all = np.concatenate([Y_train, Y_test], axis = 0)

    self.h = multi_forecasting_params['forecast_horizons']
    self.test_size = multi_forecasting_params['test_size'] #T
    self.reestimation_window = multi_forecasting_params['reestimation_window']
    self.R = int(self.test_size / self.reestimation_window)

    self.n_var = multi_forecasting_params['n_var']
    self.var_names = multi_forecasting_params['var_names']

    self.benchmarks = multi_forecasting_params['benchmarks']
    self.experiments_names = []
    self.M_varnn = multi_forecasting_params['M_varnn']

    self.Y_pred_big = None
    self.Y_pred_big_latest = None

    self._load_results()

  def _load_results(self): 

    # Load multi-forecasting results
    experiments_names = []

    # M_total x horizon x variable x bootstraps x test x re-estimation window
    Y_pred_big = np.zeros((self.M_varnn + len(self.benchmarks), 
      self.h+1, self.n_var, self.test_size, self.R))

    # Load all the betas from different experiments
    for i in range(self.M_varnn):

      if os.path.exists(f'{self.folder_path}/multi_fcast_{i}_compiled.npz') == True:
        out = np.load(f'{self.folder_path}/multi_fcast_{i}_compiled.npz')
        FCAST = out['fcast']
        experiments_names.append(f'Exp {i}')
        FCAST_nan = FCAST.copy()
        FCAST_nan[FCAST_nan == 0] = np.nan
        Y_pred = np.nanmedian(FCAST_nan, axis = 2)
        Y_pred_big[i, :,:,:,:] = Y_pred

    # Add the benchmark models in
    benchmark_folder_path = f'{self.folder_path}/benchmarks'

    for bid, benchmark in enumerate(self.benchmarks):
      FCAST = np.load(f'{benchmark_folder_path}/benchmark_multi_{benchmark}.npz')
      FCAST_nan = FCAST.copy()
      FCAST_nan[FCAST_nan == 0] = np.nan
      if benchmark in ['XGBoost', 'RF']:
        Y_pred = np.nanmedian(FCAST_nan, axis = 2)
        Y_pred_big[self.M_varnn + bid, :,:,:,:] = Y_pred
      else:
        Y_pred_big[self.M_varnn + bid, :,:,:,:] = FCAST_nan[:, :, :, 0:1]

    self.Y_pred_big = Y_pred_big
    self.Y_pred_big_latest = Y_pred_big[:, :, :, :, -1]
    self.experiments_names = experiments_names + self.benchmarks


  def plot_different_horizons(self):

    for horizon in range(1, self.h+1):
      fig, ax = plt.subplots(1,self.n_var, figsize = (self.n_var * 6, 4), constrained_layout = True)

      # Plot actual
      for var in range(self.n_var):
        ax[var].set_title(f'{self.var_names[var]}, h = {horizon}')
        ax[var].plot(self.Y_test[:, var], label = 'Actual', color = 'black')
      
      # Plot predicted
      for model in range(self.Y_pred_big_latest.shape[0]):

        Y_pred_h = np.transpose(self.Y_pred_big_latest[model, horizon, :, :]).copy()
        # Shift forward by horizon
        Y_pred_h[horizon:, :] = Y_pred_h[:(self.test_size-horizon), :]
        Y_pred_h[:horizon, :] = np.nan

        for var in range(self.n_var):
          ax[var].plot(Y_pred_h[:, var], label = self.experiments_names[model])
      if var == (self.n_var - 1):
        ax[var].legend()

    image_file = f'{self.image_folder_path}/multi_forecast_preds_diff_horizons.png'
    plt.savefig(image_file)
    print(f'Multi-forecasting Different Horizon Preds plotted at {image_file}')

  def plot_forecast_errors(self):
    for horizon in range(1, self.h+1):
      fig, ax = plt.subplots(1,self.n_var, figsize = (self.n_var * 6, 4), constrained_layout = True)

      cum_error_benchmark = np.zeros((self.test_size, self.n_var))
      cum_error_benchmark[:] = np.nan

      # Plot actual
      for var in range(self.n_var):
        ax[var].set_title(f'{self.var_names[var]}, h = {horizon}')
        
      # Plot predicted
      for model in range(self.Y_pred_big_latest.shape[0]):
        Y_pred_h = np.transpose(self.Y_pred_big_latest[model, horizon, :, :]).copy()
        # Shift forward by horizon
        Y_pred_h[horizon:, :] = Y_pred_h[:(self.test_size-horizon), :]
        Y_pred_h[:horizon, :] = np.nan

        for var in range(self.n_var):
          actual = self.Y_test[:, var]
          pred = Y_pred_h[:, var]
          error = np.abs(actual - pred)
          cum_error = np.nancumsum(error)
          
          if model == 0:
            cum_error_benchmark[:,var] = cum_error.copy()
          
          ax[var].plot(cum_error - cum_error_benchmark[:,var], label = self.experiments_names[model])

      if var == (self.n_var - 1):
        ax[var].legend()

    image_file = f'{self.image_folder_path}/multi_forecast_cum_errors.png'
    plt.savefig(image_file)
    print(f'Multi-forecasting Cum Errors plotted at {image_file}')