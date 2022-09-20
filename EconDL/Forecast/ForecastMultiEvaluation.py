import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import colorcet as cc
import seaborn as sns
import os

from sklearn.metrics import mean_absolute_error

palette = sns.color_palette(cc.glasbey, n_colors = 30)

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

    self.exclude_last = multi_forecasting_params['exclude_last']

    self.n_var = multi_forecasting_params['n_var']
    self.var_names = multi_forecasting_params['var_names']

    self.benchmarks = multi_forecasting_params['benchmarks']
    self.experiments_names = []
    self.M_varnn = multi_forecasting_params['M_varnn']
    self.normalize_errors_to_benchmark = multi_forecasting_params.get('normalize_errors_to_benchmark', True)

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

      if os.path.exists(f'{self.folder_path}/multi_fcast_params_{i}_compiled.npz') == True:
        out = np.load(f'{self.folder_path}/multi_fcast_params_{i}_compiled.npz')
        results = np.load(f'{self.folder_path}/params_{i}_compiled.npz', allow_pickle=True)['results'].item()
        experiment_name = results['params']['name']

        FCAST = out['fcast']
        experiments_names.append(experiment_name)
        FCAST_nan = FCAST.copy()
        FCAST_nan[FCAST_nan == 0] = np.nan
        Y_pred = np.nanmedian(FCAST_nan, axis = 2)
        Y_pred_big[i, :,:,:,:] = Y_pred
      else: # If there no multi-forecasting results, then just skip (add the experiment name so that self.experiments_names is the same length as self.M_varnn )
        results = np.load(f'{self.folder_path}/params_{i}_compiled.npz', allow_pickle=True)['results'].item()
        experiment_name = results['params']['name']
        experiments_names.append(f'{experiment_name} - No results')
        
    # Add the benchmark models in
    benchmark_folder_path = f'{self.folder_path}/benchmarks'

    for bid, benchmark in enumerate(self.benchmarks):

      if benchmark in ['XGBoost', 'RF']:  # Load the ML results - different as their results are saved w the ML models
          out = np.load(f'{self.folder_path}/multi_fcast_params_{benchmark}_compiled.npz')
          FCAST = out['fcast']
          FCAST_nan = FCAST.copy()
          Y_pred = np.nanmedian(FCAST_nan, axis = 2)
          Y_pred_big[self.M_varnn + bid, :,:,:,:] = Y_pred
      else:
        FCAST = np.load(f'{benchmark_folder_path}/benchmark_multi_{benchmark}.npz')
        FCAST_nan = FCAST.copy()
        #FCAST_nan[FCAST_nan == 0] = np.nan
        Y_pred_big[self.M_varnn + bid, :,:,:,:] = FCAST_nan[:, :, :, 0:1]

    self.Y_pred_big = Y_pred_big
    self.Y_pred_big_latest = Y_pred_big[:, :, :, :, -1]

    self.experiments_names = experiments_names + self.benchmarks

  def plot_different_horizons_same_model(self):
    n_models = self.Y_pred_big_latest.shape[0]
    fig, ax = plt.subplots(n_models, self.n_var, figsize = (self.n_var * 6, n_models * 4), constrained_layout = True)

    my_cmap = cm.viridis
    my_norm = colors.Normalize(vmin = 0, vmax = self.h)

    # Plot the actual in each model
    for model in range(n_models):
      
      # Plot actual
      for var in range(self.n_var):
        ax[model, var].set_title(f'{self.experiments_names[model]} - {self.var_names[var]}')
        if self.exclude_last > 0:
          ax[model, var].plot(self.Y_test[:-self.exclude_last, var], label = 'Actual', color = 'black')
        else:
          ax[model, var].plot(self.Y_test[:, var], label = 'Actual', color = 'black')
      
      # Plot predicted for each horizon
      for horizon in range(1, self.h+1):

        Y_pred_h = np.transpose(self.Y_pred_big_latest[model, horizon, :, :]).copy()
        # Shift forward by horizon
        Y_pred_h[(horizon-1):, :] = Y_pred_h[:(self.test_size-(horizon-1)), :]
        Y_pred_h[:(horizon-1), :] = np.nan

        for var in range(self.n_var):
          if self.exclude_last > 0:
            ax[model, var].plot(Y_pred_h[:-self.exclude_last, var], label = horizon, color = my_cmap(my_norm(horizon)))
          else:
            ax[model, var].plot(Y_pred_h[:, var], label = horizon, color = my_cmap(my_norm(horizon)))
      if var == (self.n_var - 1) and model == 0:
        ax[model, var].legend()

    image_file = f'{self.image_folder_path}/multi_forecast_preds_diff_horizons_each_model.png'
    plt.savefig(image_file)
    print(f'Multi-forecasting Different Horizon Each Model Preds plotted at {image_file}')


  def plot_different_horizons(self):

    fig, ax = plt.subplots(self.h, self.n_var, figsize = (self.n_var * 6, self.h * 4), constrained_layout = True)

    print('Experiments Names', self.experiments_names)
    print('Number of models', self.Y_pred_big_latest.shape[0])

    for horizon in range(1, self.h+1):
      # Plot actual
      for var in range(self.n_var):
        ax[horizon-1, var].set_title(f'{self.var_names[var]}, h = {horizon}')
        if self.exclude_last > 0:
          ax[horizon-1, var].plot(self.Y_test[:-self.exclude_last, var], label = 'Actual', color = 'black')
        else:
          ax[horizon-1, var].plot(self.Y_test[:, var], label = 'Actual', color = 'black')
      
      # Plot predicted
      for model in range(self.Y_pred_big_latest.shape[0]):

        Y_pred_h = np.transpose(self.Y_pred_big_latest[model, horizon, :, :]).copy()
        # Shift forward by horizon
        Y_pred_h[(horizon-1):, :] = Y_pred_h[:(self.test_size-(horizon-1)), :]
        Y_pred_h[:(horizon-1), :] = np.nan

        for var in range(self.n_var):
          if self.exclude_last > 0:
            ax[horizon-1, var].plot(Y_pred_h[:-self.exclude_last, var], label = self.experiments_names[model], color = palette[model], 
                                    ls = 'solid' if model < self.M_varnn else 'dotted')
          else:
            ax[horizon-1, var].plot(Y_pred_h[:, var], label = self.experiments_names[model], color = palette[model],
                                    ls = 'solid' if model < self.M_varnn else 'dotted')
      if var == (self.n_var - 1) and horizon == 1:
        ax[horizon-1, var].legend()

    image_file = f'{self.image_folder_path}/multi_forecast_preds_diff_horizons.png'
    plt.savefig(image_file)
    print(f'Multi-forecasting Different Horizon Preds plotted at {image_file}')

  def plot_forecast_errors(self):

    n_models = self.Y_pred_big_latest.shape[0]

    fig, ax = plt.subplots(self.h ,self.n_var, figsize = (self.n_var * 6, self.h * 4), constrained_layout = True)
  
    errors = np.zeros((n_models, self.test_size - self.exclude_last, self.h, self.n_var))
  
    for horizon in range(1, self.h+1):

      # Plot actual
      for var in range(self.n_var):
        ax[horizon-1, var].set_title(f'{self.var_names[var]}, h = {horizon}')
        
      # Plot predicted
      for model in range(n_models):
        Y_pred_h = np.transpose(self.Y_pred_big_latest[model, horizon, :, :]).copy()
        # Shift forward by horizon
        Y_pred_h[(horizon-1):, :] = Y_pred_h[:(self.test_size-(horizon-1)), :]
        Y_pred_h[:(horizon-1), :] = np.nan

        for var in range(self.n_var):
          if self.exclude_last > 0:
            actual = self.Y_test[:-self.exclude_last, var]
            pred = Y_pred_h[:-self.exclude_last, var]
          else:
            actual = self.Y_test[:, var]
            pred = Y_pred_h[:, var]
          error = np.abs(actual - pred)

          #if horizon == 1 and var == 1 and model == 3:
            #print('Multi-forecasting Actual', actual)
            #print('Multi-forecasting Pred', pred)
            #print('Multi-forecasting Error', error)
            

          # Store the errors in the errors array
          errors[model, :, horizon-1, var] = error

    cum_errors = np.nancumsum(errors, axis = 1)
    cum_error_benchmark = cum_errors[self.M_varnn + 1, :, :, :] # Benchmark is the AR rolling model

    # After computing all the cum errors, plot them
    for horizon in range(1, self.h+1):
      for model in range(n_models):
        for var in range(self.n_var):
          if model < self.M_varnn:
            ax[horizon-1, var].plot(cum_errors[model, :, horizon-1, var] - cum_error_benchmark[:, horizon-1, var], label = self.experiments_names[model], color = palette[model])
          else: # Dotted lines plot for benchmarks
            ax[horizon-1, var].plot(cum_errors[model, :, horizon-1, var] - cum_error_benchmark[:, horizon-1, var], label = self.experiments_names[model], color = palette[model], ls = 'dotted')

      if var == (self.n_var - 1) and horizon == 1:
        ax[horizon-1, var].legend()

    image_file = f'{self.image_folder_path}/multi_forecast_cum_errors.png'
    plt.savefig(image_file)
    print(f'Multi-forecasting Cum Errors plotted at {image_file}')

    # Calculate MAE and save file
    
    maes = np.nanmean(errors, axis = 1)

    maes_reshaped = maes.reshape(maes.shape[0] * maes.shape[1], maes.shape[2])

    mae_df = pd.DataFrame(maes_reshaped,
                columns = self.var_names
    )
    mae_df['model'] = np.repeat(self.experiments_names, maes.shape[1])
    mae_df['horizon'] = np.tile(np.arange(1, maes.shape[1] + 1), maes.shape[0])

    # Standardize errors by benchmark model
    if self.normalize_errors_to_benchmark == True:
      normalized_df = pd.DataFrame()
      for horizon in range(1, self.h+1):
        maes_horizon = mae_df.loc[mae_df['horizon'] == horizon, :].copy()
        maes_horizon[self.var_names] = maes_horizon[self.var_names] / maes_horizon.loc[maes_horizon['model'] == self.experiments_names[self.M_varnn + 1], self.var_names].values
        normalized_df = pd.concat([normalized_df, maes_horizon])
      mae_df = normalized_df

    mae_df = mae_df[['model', 'horizon'] + self.var_names]
    mae_df['model_id'] = mae_df['model'].apply(lambda x: self.experiments_names.index(x))
    mae_df = mae_df.sort_values(by = ['horizon', 'model_id'])
    mae_df = mae_df.drop(columns = ['model_id'])

    # Convert into the format in PGC's papers
    mae_df = mae_df.melt(['model', 'horizon'], var_name = 'variable', value_name = 'MAE')
    mae_df = mae_df.sort_values(['variable', 'horizon']).pivot(values = 'MAE',
                                                    index = ['variable', 'horizon'],
                                                    columns = 'model'
    )
    mae_df = mae_df[self.experiments_names]
    mae_df = mae_df.reset_index()

    mae_df.to_csv(f'{self.image_folder_path}/multi_forecast_errors.csv', index = False)