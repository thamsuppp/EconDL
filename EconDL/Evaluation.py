from calendar import c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import os
import seaborn as sns
import colorcet as cc
import matplotlib.cm as cm
import matplotlib.colors as colors
from EconDL.Forecast.ForecastMultiEvaluation import ForecastMultiEvaluation

from datetime import datetime
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scipy.stats import multivariate_normal
import pyreadr


plotly_api_key = 'Dns1gp04h4QpiskQPFT3'
chart_studio.tools.set_credentials_file(username= 'thamsuppp', api_key = plotly_api_key)

palette = sns.color_palette(cc.glasbey, n_colors = 30)

class Evaluation:
  def __init__(self, Run):

    '''
    What Run has:
    - folder_path, image_folder_path
    - n_var, var_names, test_size
    - run params: num_inner_bootstraps, n_lag_linear, n_lag_d
    - M_varnn: num_experiments
    '''

    # Run object
    self.Run = Run
    evaluation_params = self.Run.evaluation_params

    self.run_name = self.Run.run_name
    self.folder_path = self.Run.folder_path
    self.image_folder_path = self.Run.image_folder_path
    self.n_var = self.Run.n_var
    self.var_names = self.Run.var_names
    self.beta_names = ['Constant'] + self.var_names
    self.test_size = self.Run.run_params['test_size']

    self.dataset_name = self.Run.dataset_name
    self.exclude_2020 = evaluation_params.get('exclude_2020', False)
    self.first_test_id_to_exclude = None

    self.exps_to_plot = evaluation_params['exps_to_plot'] if evaluation_params['exps_to_plot'] is not None else list(range(self.Run.num_experiments))
    self.need_to_combine = evaluation_params['need_to_combine']
    self.is_simulation = evaluation_params['is_simulation']
    self.repeats_to_include = evaluation_params['repeats_to_include']
    self.is_test = evaluation_params['is_test']
    self.multiple_datasets = evaluation_params['multiple_datasets']
    self.plot_all_bootstraps = evaluation_params['plot_all_bootstraps']
    self.sim_dataset = evaluation_params['sim_dataset']
    self.benchmark_names = evaluation_params['benchmarks']
    self.test_exclude_last = evaluation_params['test_exclude_last']
    self.normalize_errors_to_benchmark = evaluation_params['normalize_errors_to_benchmark']
    self.experiments_to_compare = evaluation_params.get('experiments_to_compare', None)
    self.stoch_vol_results_name = evaluation_params.get('stoch_vol_results_name', None)

    # Store the names of every hyperparameter list
    self.experiment_names = []

    self.params = []
    self.BETAS_IN_ALL = None
    self.BETAS_ALL = None
    self.SIGMAS_IN_ALL = None
    self.SIGMAS_ALL = None
    self.PRECISION_IN_ALL = None
    self.PRECISION_ALL = None
    self.CHOLESKY_IN_ALL = None
    self.CHOLESKY_ALL = None
    self.SIGMAS_CONS_ALL = None
    self.PREDS_ALL = None
    self.PREDS_TEST_ALL = None

    self.Y_train = None
    self.Y_test = None

    if evaluation_params['experiments_to_load'] is None:
      self.M_varnn = self.Run.num_experiments
      self.experiments_to_load = list(range(self.Run.num_experiments))
    else:
      self.M_varnn = len(evaluation_params['experiments_to_load'])
      self.experiments_to_load = evaluation_params['experiments_to_load']

      self.Run.num_experiments = self.M_varnn
      experiments = []
      for i in self.experiments_to_load:
        experiments.append(self.Run.experiments[i])
      self.Run.experiments = experiments

    self.M_benchmarks = len(self.benchmark_names)
    self.M_total = self.M_varnn + self.M_benchmarks
    self.num_bootstraps = None

    self.evaluation_metrics = []


    # Load the results and params
    self.compile_results()
    self.load_results()
    if self.exclude_2020 == True:
      self.exclude_2020_results()

  def check_results_sizes(self):
    return {
      'BETAS_IN_ALL': self.BETAS_IN_ALL.shape,
      'BETAS_ALL': self.BETAS_ALL.shape,
      'SIGMAS_IN_ALL': self.SIGMAS_IN_ALL.shape,
      'SIGMAS_ALL': self.SIGMAS_ALL.shape,
      'PRECISION_IN_ALL': self.PRECISION_IN_ALL.shape,
      'PRECISION_ALL': self.PRECISION_ALL.shape,
      'CHOLESKY_IN_ALL': self.CHOLESKY_IN_ALL.shape,
      'CHOLESKY_ALL': self.CHOLESKY_ALL.shape,
      'PREDS_ALL': self.PREDS_ALL.shape,
      'PREDS_TEST_ALL': self.PREDS_TEST_ALL.shape
    }

  # Compiles all experiments' results from different repeats
  def compile_results(self):
    print(f'Evaluation compile_results: Repeats_to_include {self.repeats_to_include}')
    if self.need_to_combine == True:
      self.Run.compile_experiments(repeats_to_include = self.repeats_to_include)
      self.Run.compile_ml_experiments(repeats_to_include = self.repeats_to_include)
    else:
      print('Need to combine off, no need to compile')

  # Removes the COVID era data
  def exclude_2020_results(self):
    if self.dataset_name == 'monthly_new':
      n_obs_total = self.BETAS_ALL.shape[1]
      indices_to_exclude = [(n_obs_total + i) for i in range(-31, -19, 1)]
    elif self.dataset_name == 'quarterly_new':
      n_obs_total = self.BETAS_ALL.shape[1]
      indices_to_exclude = [(n_obs_total + i) for i in range(-10, -6, 1)]

    indices_to_include = [e for e in range(n_obs_total) if e not in indices_to_exclude]

    test_indices_to_exclude = [(e - (n_obs_total - self.test_size)) for e in indices_to_exclude if e >= (n_obs_total - self.test_size)]
    self.first_test_id_to_exclude = min(test_indices_to_exclude)
    test_indices_to_include = [(e - (n_obs_total - self.test_size)) for e in indices_to_include if e >= (n_obs_total - self.test_size)]

    # Resize these arrays to exclude the 2020 data
    self.BETAS_IN_ALL = self.BETAS_IN_ALL[:, indices_to_include, :, :, :, :]
    self.BETAS_ALL = self.BETAS_ALL[:, indices_to_include, :, :, :, :]
    self.SIGMAS_IN_ALL = self.SIGMAS_IN_ALL[:, indices_to_include, :, :, :]
    self.SIGMAS_ALL = self.SIGMAS_ALL[:, indices_to_include, :, :, :]
    self.PRECISION_IN_ALL = self.PRECISION_IN_ALL[:, indices_to_include, :, :, :]
    self.PRECISION_ALL = self.PRECISION_ALL[:, indices_to_include, :, :, :]
    self.CHOLESKY_IN_ALL = self.CHOLESKY_IN_ALL[:, indices_to_include, :, :, :, :]
    self.CHOLESKY_ALL = self.CHOLESKY_ALL[:, indices_to_include, :, :, :, :]
    self.PREDS_TEST_ALL = self.PREDS_TEST_ALL[:, test_indices_to_include, :, :] # Only need to resize the test set
    self.Y_test = self.Y_test[test_indices_to_include, :]
      

  # Loads the compiled results and benchmarks into the object
  def load_results(self):

    # Load VARNN results
    for i in range(self.M_varnn):
      experiment = self.experiments_to_load[i]

      compiled_text = 'compiled' if self.need_to_combine == True else 'repeat_0'
      dataset_text = f'_dataset_{self.sim_dataset}' if self.multiple_datasets == True else ''
      load_file = f'{self.folder_path}/params_{experiment}{dataset_text}_{compiled_text}.npz'

      print(f'Evaluation load_results(): load_file: {load_file}')

      results = np.load(load_file, allow_pickle = True)['results'].item()

      params = results['params']
      self.params.append(params)

      n_lag_linear = params['n_lag_linear']
      num_bootstraps = params['num_bootstrap']
      
      self.experiment_names.append(params['name'])
      BETAS = results['betas']
      BETAS_IN = results['betas_in']
      SIGMAS = results['sigmas']
      SIGMAS_IN = results['sigmas_in']
      PRECISION = results['precision']
      PRECISION_IN = results['precision_in']
      CHOLESKY = results['cholesky']
      CHOLESKY_IN = results['cholesky_in']
      PREDS = results['train_preds']
      PREDS_TEST = results['test_preds']
      

      # Estimate time-invariant cov mat from the residuals
      Y_train = results['y']
      Y_test = results['y_test']
      resids = np.repeat(np.expand_dims(Y_train, axis = 1), PREDS.shape[1], axis = 1) - PREDS

      # For experiments with more than 1 lag, get the ids of the 1st beta to plot
      beta_ids_to_keep = [0] + list(range(1, BETAS_IN.shape[1], n_lag_linear))
      BETAS_IN = BETAS_IN[:, beta_ids_to_keep, :,:,:]
      BETAS = BETAS[:, beta_ids_to_keep, :,:,:]

      n_hemis = CHOLESKY.shape[3] # no longer betas as there is the mean hemisphere

      if i == 0:
        BETAS_ALL = np.zeros((self.M_total, BETAS.shape[0], BETAS.shape[1], BETAS.shape[2], BETAS.shape[3], 4))
        #BETAS_ALL = np.zeros((self.M_total, BETAS.shape[0], BETAS.shape[1], BETAS.shape[2], BETAS.shape[3], BETAS.shape[4]))
        BETAS_ALL[:] = np.nan
        # n_models x n_obs x n_betas x n_bootstraps x n_vars x n_hemispheres
        BETAS_IN_ALL = np.zeros((self.M_total, BETAS_IN.shape[0], BETAS_IN.shape[1], BETAS.shape[2], BETAS_IN.shape[3], 4))
        #BETAS_IN_ALL = np.zeros((self.M_total, BETAS_IN.shape[0], BETAS_IN.shape[1], BETAS.shape[2], BETAS_IN.shape[3], BETAS_IN.shape[4]))
        BETAS_IN_ALL[:] = np.nan 

        # n_models x n_obs x n_vars x n_vars x n_bootstraps
        SIGMAS_ALL = np.zeros((self.M_total, SIGMAS.shape[0], SIGMAS.shape[1], SIGMAS.shape[2], SIGMAS.shape[3]))
        SIGMAS_ALL[:] = np.nan
        PRECISION_ALL = np.zeros_like(SIGMAS_ALL)
        PRECISION_ALL[:] = np.nan
        CHOLESKY_ALL = np.zeros((self.M_total, SIGMAS.shape[0], SIGMAS.shape[1], SIGMAS.shape[2], 3, SIGMAS.shape[3]))
        CHOLESKY_ALL[:] = np.nan 

        SIGMAS_IN_ALL = np.zeros((self.M_total, SIGMAS_IN.shape[0], SIGMAS_IN.shape[1], SIGMAS_IN.shape[2], SIGMAS_IN.shape[3]))
        SIGMAS_IN_ALL[:] = np.nan 
        PRECISION_IN_ALL = np.zeros_like(SIGMAS_IN_ALL)
        PRECISION_IN_ALL[:] = np.nan
        CHOLESKY_IN_ALL = np.zeros_like(CHOLESKY_ALL)
        CHOLESKY_IN_ALL[:] = np.nan

        SIGMAS_CONS_ALL = np.zeros((self.M_total, SIGMAS.shape[1], SIGMAS.shape[2], SIGMAS.shape[3]))
        SIGMAS_CONS_ALL[:] = np.nan

        PREDS_ALL = np.zeros((self.M_total, PREDS.shape[0], PREDS.shape[1], PREDS.shape[2]))
        PREDS_ALL[:] = np.nan
        PREDS_TEST_ALL = np.zeros((self.M_total, PREDS_TEST.shape[0], PREDS_TEST.shape[1], PREDS_TEST.shape[2]))
        PREDS_TEST_ALL[:] = np.nan 

        self.num_bootstraps = BETAS.shape[2]

      # If >1 hemis, Demean the time hemisphere and add the mean to the endogenous hemisphere
      # (note: the means are the in-sample means, not the oob ones)
      # if BETAS.shape[4] > 1:
      #   time_hemi_means = np.nanmean(BETAS_IN[:,:,:,:,1], axis = 0)
      #   time_hemi_means_expand = np.repeat(np.expand_dims(time_hemi_means, axis = 0), BETAS.shape[0], axis = 0)
        # BETAS_IN[:, :, :, :, 0] = BETAS_IN[:, :, :, :, 0] + time_hemi_means_expand
        # BETAS_IN[:, :, :, :, 1] = BETAS_IN[:, :, :, :, 1] - time_hemi_means_expand
        # BETAS[:, :, :, :, 0] = BETAS[:, :, :, :, 0] + time_hemi_means_expand
        # BETAS[:, :, :, :, 1] = BETAS[:, :, :, :, 1] - time_hemi_means_expand

      BETAS_ALL[i,:,:,:,:, :(n_hemis+1)] = BETAS
      BETAS_IN_ALL[i,:,:,:,:, :(n_hemis+1)] = BETAS_IN
      SIGMAS_ALL[i, :,:,:,:] = SIGMAS
      SIGMAS_IN_ALL[i, :,:,:,:] = SIGMAS_IN
      PRECISION_ALL[i, :,:,:,:] = PRECISION
      PRECISION_IN_ALL[i, :,:,:,:] = PRECISION_IN
      CHOLESKY_ALL[i, :,:,:, :n_hemis, :] = CHOLESKY
      CHOLESKY_IN_ALL[i, :,:,:, :n_hemis, :] = CHOLESKY_IN
      PREDS_ALL[i,:,:,:] = PREDS
      PREDS_TEST_ALL[i,:,:,:] = PREDS_TEST

      for b in range(num_bootstraps):
        SIGMAS_CONS_ALL[i, :,:,b] = pd.DataFrame(resids[:, b, :]).dropna().cov()
        

    self.BETAS_ALL = BETAS_ALL
    self.BETAS_IN_ALL = BETAS_IN_ALL
    self.SIGMAS_IN_ALL = SIGMAS_IN_ALL
    self.SIGMAS_ALL = SIGMAS_ALL
    self.PRECISION_IN_ALL = PRECISION_IN_ALL
    self.PRECISION_ALL = PRECISION_ALL
    self.CHOLESKY_IN_ALL = CHOLESKY_IN_ALL
    self.CHOLESKY_ALL = CHOLESKY_ALL
    self.SIGMAS_CONS_ALL = SIGMAS_CONS_ALL
    self.PREDS_ALL = PREDS_ALL
    self.PREDS_TEST_ALL = PREDS_TEST_ALL
    self.Y_train = Y_train
    self.Y_test = Y_test
    
    # Shift the means of the BETAS
    #self._shift_betas_means()

    # Load the benchmarks
    self._load_benchmarks()

    # Update all_names
    self.all_names = self.experiment_names + self.benchmark_names
    
  # Normalize such that all have mean of 0 (create mean hemisphere)
  # def _shift_betas_means(self):
    
  #   print('_shift_betas_means: Test size', self.test_size)
  #   # Take the training OOB mean of the betas
  #   BETAS_ALL_MEANS = np.nanmean(self.BETAS_ALL[:, :(-self.test_size), :, :, :, :], axis = 1)
    
  #   #np.save('BETAS_ALL.npy', self.BETAS_ALL)
    
  #   print('_shift_betas_means: BETAS_ALL_MEANS.shape = ', BETAS_ALL_MEANS.shape)
  #   # n_models x n_betas x n_bootstraps x n_vars x n_hemispheres
  #   BETAS_ALL_MEANS_expand = np.repeat(np.expand_dims(BETAS_ALL_MEANS, axis = 1), self.BETAS_ALL.shape[1], axis = 1)
  #   print('_shift_betas_means: BETAS_ALL_MEANS_expand.shape = ', BETAS_ALL_MEANS_expand.shape)
  #   print('_shift_betas_means: self.BETAS_ALL.shape = ', self.BETAS_ALL.shape)
  #   # Remove training OOB mean from BETAS_ALL
  #   self.BETAS_ALL = self.BETAS_ALL - BETAS_ALL_MEANS_expand
  #   self.BETAS_MEAN_ALL = BETAS_ALL_MEANS # n_betas x n_bootstraps x n_vars x n_hemispheres
  #   # Remove training OOB mean from BETAS_ALL
  #   self.BETAS_IN_ALL = self.BETAS_IN_ALL - BETAS_ALL_MEANS_expand
    

  def _load_benchmarks(self):
    
    benchmark_folder_path = f'{self.folder_path}/benchmarks'

    if os.path.isdir(benchmark_folder_path) == False:
      print('Evaluation _load_benchmarks(): No benchmarks folder')
      return

    for i in range(self.M_benchmarks):
      print(f'Starting to load {i}: {self.benchmark_names[i]}')
      out = np.load(f'{benchmark_folder_path}/benchmark_{self.benchmark_names[i]}.npz')

      preds = out['train_preds']
      preds_test = out['test_preds']

      preds = np.repeat(np.expand_dims(preds, axis = 1), self.num_bootstraps, axis = 1)
      preds_test = np.repeat(np.expand_dims(preds_test, axis = 1), self.num_bootstraps, axis = 1)
      self.PREDS_ALL[self.M_varnn + i, :,:,:] = preds
      self.PREDS_TEST_ALL[self.M_varnn + i,:,:,:] = preds_test
      
  def evaluate_predictive_density(self):
        
    ### Evaluate the predictive density for StochVol Benchmark
    
    arfit = pyreadr.read_r(f'data/stochvol_results/arfit_{self.stoch_vol_results_name}.RData')['arfit_all'].to_numpy().T
    svfit = pyreadr.read_r(f'data/stochvol_results/svfit_{self.stoch_vol_results_name}.RData')['svfit_all'].to_numpy().T
    # Stochvol loses 2 observations at the beginning
    Y_test = self.Y_test[2:, :]
    
    # Plot y and the predicted mean in a graph (testing purposes)
    # for var_num in range(Y_test.shape[1]):
    #   y = Y_test[:,var_num]
    #   ar = arfit[:,var_num]
    #   sv = svfit[:, var_num]
      
    #   var_name = self.var_names[var_num]
      
    #   # Create a 2x1 subplot, plotting arfit_all and y in the first plot, and svfit_all in the second plot
    #   fig, (ax1, ax2) = plt.subplots(2, 1)
    #   fig.suptitle('SV Fit for ' + var_name)
    #   ax1.plot(ar, label='AR(2) Mean Fit')
    #   ax1.plot(y, label='Actual')
    #   ax1.legend()
    #   ax2.plot(sv, label='SVol Fit')
    #   ax2.legend()
    #   plt.show()
    #   image_file = f"{self.image_folder_path}/stochvol_density_{var_name}.png"
    #   plt.savefig(image_file)
    #   plt.close()
    if self.test_exclude_last > 0:
      preds_test_mean = arfit[:-self.test_exclude_last, :]
      sigmas_mean = svfit[:-self.test_exclude_last, :]
      Y_test = Y_test[:-self.test_exclude_last, :]
    else:
      preds_test_mean = arfit
      sigmas_mean = svfit
    
    PRED_DENSITY_MARG_SVOL = np.zeros((1, preds_test_mean.shape[0] + 2, preds_test_mean.shape[1]))
    PRED_DENSITY_MARG_SVOL[:] = np.nan
    # Loop over all time steps
    for t in range(preds_test_mean.shape[0]):
      pred_mean = preds_test_mean[t, :]
      pred_sigma = sigmas_mean[t, :]
      y_test = Y_test[t, :]
      
      for var in range(preds_test_mean.shape[1]):
        # Construct a univariate normal with pred_mean and pred_sigma
        univ_norm = multivariate_normal(pred_mean[var], pred_sigma[var])
        # Evaluate density at y_test
        log_density = univ_norm.logpdf(y_test[var])
        PRED_DENSITY_MARG_SVOL[0, t+2, var] = log_density
        
    # Make dataframe wtih sum and mean of predictive density as the 2 columns
    marginal_density_df_svol = pd.DataFrame()
    for var in range(preds_test_mean.shape[1]):
      marginal_density_df_var = pd.DataFrame({'Mean': PRED_DENSITY_MARG_SVOL[:, 2:, var].mean(axis=1),
                            'Median': np.median(PRED_DENSITY_MARG_SVOL[:, 2:, var], axis=1),
                            '10th': np.percentile(PRED_DENSITY_MARG_SVOL[:, 2:, var], 10, axis=1),
                            '90th': np.percentile(PRED_DENSITY_MARG_SVOL[:, 2:, var], 90, axis=1),
                            'Variable': self.var_names[var],
                            'Experiment': 'StochVol'
                            })
      
      marginal_density_df_svol = pd.concat([marginal_density_df_svol, marginal_density_df_var], axis=0).reset_index(drop=True)
    
    marginal_density_df_svol.to_csv(f'{self.image_folder_path}/marginal_density_svol_test.csv')
    
  
    ### Evaluate the predictive density for VARNN
        
    PRED_DENSITY_MARG_ALL = np.zeros((self.M_varnn, self.PREDS_TEST_ALL.shape[1], self.PREDS_TEST_ALL.shape[3]))
    PRED_DENSITY_JOINT_ALL = np.zeros((self.M_varnn, self.PREDS_TEST_ALL.shape[1]))

    for model in range(self.M_varnn):
      
      PREDS_TEST = self.PREDS_TEST_ALL[model, :,:,:] # n_obs x n_bootstraps x n_vars
      SIGMAS = self.SIGMAS_ALL[model, -self.test_size:,:,:,:] # n_obs x n_vars x n_vars x n_bootstraps

      # Take the mean across all bootstraps: 20 x 3
      preds_test_mean = PREDS_TEST.mean(axis=1)
      # Take the mean of all cov mats across all bootstraps. sigmas_mean: 20 x 3 x 3
      sigmas_mean = SIGMAS.mean(axis = 3)

      # Loop over all time steps
      for t in range(PREDS_TEST.shape[0]):
        pred_mean = preds_test_mean[t, :]
        pred_sigma = sigmas_mean[t, :, :]
        y_test = self.Y_test[t, :]
        
        ### Calculating joint density
        # Construct a multivariate normal with pred_mean and pred_sigma
        mv_norm = multivariate_normal(pred_mean, pred_sigma)
        # Evaluate the density at y_test
        log_density = mv_norm.logpdf(y_test)
        PRED_DENSITY_JOINT_ALL[model, t] = log_density
        
        ### Calculating marginal density
        for var in range(PREDS_TEST.shape[2]):
          # Construct a univariate normal with pred_mean and pred_sigma
          univ_norm = multivariate_normal(pred_mean[var], pred_sigma[var, var])
          # Evaluate the density at y_test
          log_density = univ_norm.logpdf(y_test[var])
          PRED_DENSITY_MARG_ALL[model, t, var] = log_density
          
    ### Plot the predictive density of all the models' predictive densities pre-COVID
    fig, axs = plt.subplots(self.M_varnn + 1, 1, figsize = (6, 4 * self.M_varnn + 1), constrained_layout=True)
    for model in range(self.M_varnn):
      for var in range(self.n_var):
        if self.test_exclude_last > 0:
          axs[model].plot(PRED_DENSITY_MARG_ALL[model, :-self.test_exclude_last, var], label = self.var_names[var])
        else:
          axs[model].plot(PRED_DENSITY_MARG_ALL[model, :, var], label = self.var_names[var])  
      # Place title (experiment name)
      axs[model].set_title(self.experiment_names[model])
      # Place legend on the first figure
      if model == 0:
        axs[model].legend()
        
    # Plot the predictive density for SVol benchmark
    for var in range(self.n_var):
      axs[-1].plot(PRED_DENSITY_MARG_SVOL[0, :, var], label = self.var_names[var])
    axs[-1].set_title('Stochastic Volatility Benchmark')
    
    image_file = f"{self.image_folder_path}/pred_density_by_model.png"
    plt.savefig(image_file)
    plt.close()
    
    # Make dataframe wtih sum and mean of predictive density as the 2 columns
    joint_density_df = pd.DataFrame({'Mean': PRED_DENSITY_JOINT_ALL.mean(axis=1), 
                                     'Median': np.median(PRED_DENSITY_JOINT_ALL, axis=1),
                      '10th': np.percentile(PRED_DENSITY_JOINT_ALL, 10, axis=1),
                      '90th': np.percentile(PRED_DENSITY_JOINT_ALL, 90, axis=1)
    }, index = self.experiment_names)
    
    ### Plot the predictive density for each variable
    fig, axs = plt.subplots(self.n_var, 1, figsize = (6, 4 * self.n_var), constrained_layout=True)
    for var in range(self.n_var):
      for model in range(self.M_varnn):
        if self.test_exclude_last > 0:
          axs[var].plot(PRED_DENSITY_MARG_ALL[model, :-self.test_exclude_last, var], label = self.experiment_names[model])
        else:
          axs[var].plot(PRED_DENSITY_MARG_ALL[model, :, var], label = self.experiment_names[model])
      # Plot the SVol bench mark
      axs[var].plot(PRED_DENSITY_MARG_SVOL[0, :, var], label = 'SVol')
      # Place title (variable name)
      axs[var].set_title(self.var_names[var])
      # Place legend on first figure
      if var == 0:
        axs[var].legend()
    
    image_file = f"{self.image_folder_path}/pred_density_by_var.png"
    plt.savefig(image_file)
    plt.close()

    # Make dataframe wtih sum and mean of predictive density as the 2 columns
    # *to make this comparable with StochVol, we need to exclude the first 2 time steps
    marginal_density_df = pd.DataFrame()
    for var in range(self.n_var):
      marginal_density_df_var = pd.DataFrame({'Mean': PRED_DENSITY_MARG_ALL[:, 2:, var].mean(axis=1),
                            'Median': np.median(PRED_DENSITY_MARG_ALL[:, 2:, var], axis=1),
                            '10th': np.percentile(PRED_DENSITY_MARG_ALL[:, 2:, var], 10, axis=1),
                            '90th': np.percentile(PRED_DENSITY_MARG_ALL[:, 2:, var], 90, axis=1),
                            'Variable': self.var_names[var]
                            }, index = self.experiment_names)
      marginal_density_df_var = marginal_density_df_var.reset_index()
      marginal_density_df_var = marginal_density_df_var.rename(columns = {'index': 'Experiment'})
      
      marginal_density_df = pd.concat([marginal_density_df, marginal_density_df_var], axis=0)
      
    marginal_density_df = pd.concat([marginal_density_df, marginal_density_df_svol], axis = 0)
    # Sort by variable
    marginal_density_df = marginal_density_df.sort_values(by = ['Variable', 'Experiment'])
    
    joint_density_df.to_csv(f'{self.image_folder_path}/joint_density_test.csv')
    marginal_density_df.to_csv(f'{self.image_folder_path}/marginal_density_test.csv')
    
    # If exclude_2020, then also output the density scores pre and post COVID (as well as the total errors)
    if self.exclude_2020 == True:
      
      ### PRE-COVID
      # Make dataframe wtih sum and mean of predictive density as the 2 columns
      joint_density_df = pd.DataFrame({'Mean': PRED_DENSITY_JOINT_ALL[:, :self.first_test_id_to_exclude].mean(axis=1), 
                        'Median': np.median(PRED_DENSITY_JOINT_ALL[:, :self.first_test_id_to_exclude], axis = 1),
                        '10th': np.percentile(PRED_DENSITY_JOINT_ALL[:, :self.first_test_id_to_exclude], 10, axis=1),
                        '90th': np.percentile(PRED_DENSITY_JOINT_ALL[:, :self.first_test_id_to_exclude], 90, axis=1)
      }, index = self.experiment_names)

      # Make dataframe wtih sum and mean of predictive density as the 2 columns
      marginal_density_df = pd.DataFrame()
      for var in range(self.n_var):
        marginal_density_df_var = pd.DataFrame({'Mean': PRED_DENSITY_MARG_ALL[:, :self.first_test_id_to_exclude, var].mean(axis=1),
                              'Median': np.median(PRED_DENSITY_MARG_ALL[:, :self.first_test_id_to_exclude, var], axis = 1),
                              '10th': np.percentile(PRED_DENSITY_MARG_ALL[:, :self.first_test_id_to_exclude, var], 10, axis=1),
                              '90th': np.percentile(PRED_DENSITY_MARG_ALL[:, :self.first_test_id_to_exclude, var], 90, axis=1),
                              'Variable': self.var_names[var]
                              }, index = self.experiment_names)
        marginal_density_df_var = marginal_density_df_var.reset_index()
        marginal_density_df_var = marginal_density_df_var.rename(columns = {'index': 'Experiment'})
        
        marginal_density_df = pd.concat([marginal_density_df, marginal_density_df_var], axis=0)
      
      joint_density_df.to_csv(f'{self.image_folder_path}/joint_density_test_pre_covid.csv')
      marginal_density_df.to_csv(f'{self.image_folder_path}/marginal_density_test_pre_covid.csv')
      
      ### POST-COVID
      # Make dataframe wtih sum and mean of predictive density as the 2 columns
      joint_density_df = pd.DataFrame({'Mean': PRED_DENSITY_JOINT_ALL[:, self.first_test_id_to_exclude:].mean(axis=1), 
                        'Median': np.median(PRED_DENSITY_JOINT_ALL[:, self.first_test_id_to_exclude:], axis = 1),
                        '10th': np.percentile(PRED_DENSITY_JOINT_ALL[:, self.first_test_id_to_exclude:], 10, axis=1),
                        '90th': np.percentile(PRED_DENSITY_JOINT_ALL[:, self.first_test_id_to_exclude:], 90, axis=1)
    }, index = self.experiment_names)

      # Make dataframe wtih sum and mean of predictive density as the 2 columns
      marginal_density_df = pd.DataFrame()
      for var in range(self.n_var):
        marginal_density_df_var = pd.DataFrame({'Mean': PRED_DENSITY_MARG_ALL[:, self.first_test_id_to_exclude:, var].mean(axis=1),
                              'Median': np.median(PRED_DENSITY_MARG_ALL[:, self.first_test_id_to_exclude:, var], axis = 1),
                              '10th': np.percentile(PRED_DENSITY_MARG_ALL[:, self.first_test_id_to_exclude:, var], 10, axis=1),
                              '90th': np.percentile(PRED_DENSITY_MARG_ALL[:, self.first_test_id_to_exclude:, var], 90, axis=1),
                              'Variable': self.var_names[var]
                              }, index = self.experiment_names)
        marginal_density_df_var = marginal_density_df_var.reset_index()
        marginal_density_df_var = marginal_density_df_var.rename(columns = {'index': 'Experiment'})
        
        marginal_density_df = pd.concat([marginal_density_df, marginal_density_df_var], axis=0)
        
      
      joint_density_df.to_csv(f'{self.image_folder_path}/joint_density_test_post_covid.csv')
      marginal_density_df.to_csv(f'{self.image_folder_path}/marginal_density_test_post_covid.csv')

    
  # Helper function to plot betas
  def _plot_betas_inner(self, BETAS, var_names, beta_names, image_file, q = 0.16, title = '', actual = None):

    # Check if the entire numpy array is nan
    if np.all(np.isnan(BETAS)):
      return

    n_obs = BETAS.shape[0]
    n_betas = BETAS.shape[1]
    n_bootstraps = BETAS.shape[2]
    n_vars = BETAS.shape[3]
    fig, axs = plt.subplots(n_vars, n_betas, figsize = (6 * n_betas, 4 * n_vars), constrained_layout = True)

    for var in range(n_vars):
      for beta in range(n_betas):

        # Get the quantiles
        betas_lcl = np.nanquantile(BETAS[:, beta, :, var], q = q, axis = 1)
        betas_ucl = np.nanquantile(BETAS[:, beta, :, var], q = 1 - q, axis = 1)
        betas_median = np.nanmean(BETAS[:, beta, :, var], axis = 1)

        # betas_median = pd.Series(betas_median, index = dates)
        # betas_lcl = pd.Series(betas_lcl, index = dates)
        # betas_ucl = pd.Series(betas_ucl, index = dates)

        #Plot all bootstraps' paths
        if self.plot_all_bootstraps == True:
          for i in range(n_bootstraps):
            axs[var, beta].plot(BETAS[:, beta, i, var], lw = 0.5, alpha = 0.15)

        axs[var, beta].plot(betas_median, label = f'{var_names[var]} {beta_names[beta]}', lw = 1.5)
        # Plot the confidence bands
        axs[var, beta].fill_between(list(range(BETAS.shape[0])), betas_lcl, betas_ucl, alpha = 0.5)

        # # Plot actual
        # if actual is not None:
        #   actual_swapped = actual.copy()[:, :, [3,0,1,2]]
        #   axs[var, beta].plot(actual_swapped[:, var, beta], color = 'black')

        # Plot the confidence bands (old method, less preferred by PGC)
        # axs[var, beta].plot(betas_lcl, label = f'{var_names[var]} {beta_names[beta]}', lw = 1.5, color = 'black', ls = '--')
        # axs[var, beta].plot(betas_ucl, label = f'{var_names[var]} {beta_names[beta]}', lw = 1.5, color = 'black', ls = '--')

        # Vertical line for train/test split
        #axs[var, beta].axvline(results['oos_index'][0], color = 'black', linewidth = 1, linestyle = '--')
        # Horizontal line for OLS estimation
        #axs[var, beta].axhline(coefs_matrix[var, beta], color = 'green', linewidth = 1)

        #axs[var, beta].set_xticks(dates)

        #Set the y-axis limits to be at the min 10% LCL and max 10% UCL
        axs[var, beta].set_ylim(
            np.nanmin(np.nanquantile(BETAS[:, beta, :, var], axis = -1, q = 0.1)),
            np.nanmax(np.nanquantile(BETAS[:, beta, :, var], axis = -1, q = 0.9))
        )

        axs[var, beta].set_title(f'{var_names[var]}, {beta_names[beta]}')
        axs[var, beta].set_xlabel('Time')
        axs[var, beta].set_ylabel('Coefficient')

    fig.suptitle(title, fontsize=16)
    plt.savefig(image_file)
    plt.close()

    print(f'Betas plotted at {image_file}')

  def plot_betas(self, is_test = False):
    # Plot individual hemisphere and summed betas
    if is_test == False:
      BETAS_ALL_PLOT = self.BETAS_IN_ALL[:, :-self.test_size,:,:,:,:]
      #coefs_tv_plot = coefs_tv[:-test_size, :, :] if is_simulation == True else None
    else:
      BETAS_ALL_PLOT = self.BETAS_ALL[:, -self.test_size:,:,:,:, :]
      #coefs_tv_plot = coefs_tv[-test_size:, :, :] if is_simulation == True else None

    for i in self.exps_to_plot:

      BETAS_EXP_PLOT = BETAS_ALL_PLOT[i, :, :, :, :, : ]
      #BETAS_MEAN_EXP_PLOT = np.nansum(self.BETAS_MEAN_ALL[i, :, :, :, :], axis = -1) # Sum the means across all the hemis

      n_hemis = 0
      for hemi in range(BETAS_ALL_PLOT.shape[5]):
        # Check that the entire array is not nan
        if np.isnan(BETAS_EXP_PLOT[:, :, :, :, hemi]).all() == False:
          n_hemis += 1

      BETAS_EXP_PLOT = BETAS_EXP_PLOT[:, :, :, :, :n_hemis]
      print(f'Experiment {i} has {n_hemis} hemispheres')

      for hemi in range(n_hemis):
        image_file = f"{self.image_folder_path}/betas_{i}_hemi_{hemi}{'_test' if is_test == True else ''}.png"
        self._plot_betas_inner(BETAS_EXP_PLOT[:, :, :, :, hemi], self.var_names, self.beta_names, image_file, q = 0.16, title = f'Experiment {i} ({self.experiment_names[i]}) Betas, Hemisphere {hemi}', actual = None)
      
      # betas_mean_exp_plot_repeat = np.repeat(np.expand_dims(BETAS_MEAN_EXP_PLOT, axis = 0), BETAS_EXP_PLOT.shape[0], axis = 0)
      # print('betas_mean_exp_plot_repeat', betas_mean_exp_plot_repeat.shape)
      # #np.save('betas_mean_exp_plot_repeat.npy', betas_mean_exp_plot_repeat)
      # # Plot the mean hemisphere
      # image_file = f"{self.image_folder_path}/betas_{i}_mean{'_test' if is_test == True else ''}.png"
      # self._plot_betas_inner(betas_mean_exp_plot_repeat, 
      #                        self.var_names, self.beta_names, image_file, q = 0.16, title = f'Experiment {i} ({self.experiment_names[i]}) Betas, Mean Hemisphere', actual = None)
        
      
      # Plot the summed betas from all hemispheres (need to add the mean back in)
      #BETAS_EXP_PLOT_SUM = np.sum(BETAS_EXP_PLOT, axis = -1) + betas_mean_exp_plot_repeat
      
      image_file = f"{self.image_folder_path}/betas_{i}_sum{'_test' if is_test == True else ''}.png"
      self._plot_betas_inner(np.sum(BETAS_EXP_PLOT, axis = -1), self.var_names, self.beta_names, image_file, q = 0.16, title = f'Experiment {i} ({self.experiment_names[i]}) Betas, Sum', actual = None)

      # @evalmetric
      # BETAS_EXP_PLOT: obs x betas x bootstraps x vars x hemis
      BETAS_EXP = np.sum(BETAS_EXP_PLOT, axis = -1)
      BETAS_EXP_MEDIAN = np.nanmedian(BETAS_EXP, axis = 2) # obs x betas x vars

      const_vol = np.nanstd(BETAS_EXP_MEDIAN[:, 0, :], axis = 0) # vars
      betas_vol = np.nanstd(BETAS_EXP_MEDIAN[:, 1:, :], axis = 0) # betas x vars

      self.evaluation_metrics.append({'metric': 'const_vol', 'experiment': i, 'value': const_vol})
      self.evaluation_metrics.append({'metric': 'betas_vol', 'experiment': i, 'value': betas_vol})

  def plot_betas_comparison(self, exps_to_compare = [0, 1], is_test = False):
    try:
      if is_test == False:
        BETAS_ALL_PLOT = self.BETAS_IN_ALL[:, :-self.test_size,:,:,:,:]
      else:
        BETAS_ALL_PLOT = self.BETAS_ALL[:, -self.test_size:,:,:,:, :]

      BETAS_COMPARE = np.zeros((len(exps_to_compare), BETAS_ALL_PLOT.shape[1], BETAS_ALL_PLOT.shape[2], BETAS_ALL_PLOT.shape[3], BETAS_ALL_PLOT.shape[4]))
      for i, exp in enumerate(exps_to_compare):
        BETAS_EXP_PLOT = BETAS_ALL_PLOT[exp, :, :, :, :]
        n_hemis = 0
        for hemi in range(BETAS_ALL_PLOT.shape[5]):
          # Check that the entire array is not nan
          if np.isnan(BETAS_EXP_PLOT[:, :, :, :, hemi]).all() == False:
            n_hemis += 1

        BETAS_EXP_PLOT = BETAS_EXP_PLOT[:, :, :, :, :n_hemis]
        BETAS_SUM = np.sum(BETAS_EXP_PLOT, axis = -1)
        BETAS_COMPARE[i, :, :, :, :] = BETAS_SUM
        
      n_obs = BETAS_COMPARE.shape[1]
      n_betas = BETAS_COMPARE.shape[2]
      n_vars = BETAS_COMPARE.shape[4]

      fig, axs = plt.subplots(n_vars, n_betas, figsize = (6 * n_betas, 4 * n_vars), constrained_layout = True)

      for var in range(n_vars):
        for beta in range(n_betas):
          for i, exp in enumerate(exps_to_compare):
            axs[var, beta].plot(np.nanmedian(BETAS_COMPARE[i, :, beta, :, var], axis = 1), label = f'{self.experiment_names[exp]}')
            axs[var, beta].fill_between(
                np.arange(n_obs),
                np.nanquantile(BETAS_COMPARE[i, :, beta, :, var], axis = 1, q = 0.16),
                np.nanquantile(BETAS_COMPARE[i, :, beta, :, var], axis = 1, q = 0.84),
                alpha = 0.5
            )
            axs[var, beta].set_title(f'{self.var_names[var]}, {self.beta_names[beta]}')
            axs[var, beta].set_xlabel('Time')
            axs[var, beta].set_ylabel('Coefficient')

            if var == 0 and beta == 0:
              axs[var, beta].legend()
      
      fig.suptitle(f'Comparison of Betas', fontsize=16)
      image_file = f"{self.image_folder_path}/betas_comparison{'_test' if is_test == True else ''}.png"
      plt.savefig(image_file)
      plt.close()
    except:
      pass

  def plot_precision(self, is_test = False):

      # Don't show test (change this code to show in-sample)
    if is_test == False:
      PRECISION_ALL_PLOT = self.PRECISION_ALL[:, :-self.test_size,:,:,:]
    else:
      PRECISION_ALL_PLOT = self.PRECISION_ALL[:, -self.test_size:,:,:,:]

    for i in self.exps_to_plot:

      # If all values of precision_all_plot are nan, skip
      if np.isnan(PRECISION_ALL_PLOT[i, :, :, :, :]).all() == True:
        continue

      fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (6 * self.n_var, 4 * self.n_var), constrained_layout = True)

      for row in range(self.n_var):
        for col in range(self.n_var):
          
          # Plot every bootstrap's value
          if self.plot_all_bootstraps == True:
            for b in range(PRECISION_ALL_PLOT.shape[4]):
              axs[row, col].plot(PRECISION_ALL_PLOT[i, :, row, col, b], lw = 0.5, alpha = 0.25, label = i)

          axs[row, col].plot(np.nanmedian(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1), color = 'black')
          axs[row, col].set_title(f'{self.var_names[row]}, {self.var_names[col]}')
          axs[row, col].set_xlabel('Time')
          axs[row, col].set_ylabel('Coefficient')

          sigmas_lcl = np.nanquantile(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.16)
          sigmas_ucl = np.nanquantile(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.84)
          axs[row, col].fill_between(list(range(PRECISION_ALL_PLOT.shape[1])), sigmas_lcl, sigmas_ucl, alpha = 0.8)

          # Set the y-axis limits to be at the min 10% LCL and max 10% UCL
          axs[row, col].set_ylim(
              np.nanmin(np.nanquantile(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.1)),
              np.nanmax(np.nanquantile(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.9))
          )

      fig.suptitle(f'Experiment {i} ({self.experiment_names[i]}) Precision', fontsize=16)
      image_file = f"{self.image_folder_path}/precision_{i}{'_test' if is_test == True else ''}.png"
      plt.savefig(image_file)
      plt.close()

      print(f'Precision plotted at {image_file}')

  def plot_cholesky(self, is_test = False):
    if is_test == False:
      CHOLESKY_ALL_PLOT = self.CHOLESKY_ALL[:, :-self.test_size,:,:,:,:]
    else:
      CHOLESKY_ALL_PLOT = self.CHOLESKY_ALL[:, -self.test_size:,:,:,:,:]

    for i in self.exps_to_plot:

      # If all values of precision_all_plot are nan, skip
      if np.isnan(CHOLESKY_ALL_PLOT[i, :, :, :, :]).all() == True:
        continue

      fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (6 * self.n_var, 4 * self.n_var), constrained_layout = True)

      for row in range(self.n_var):
        for col in range(self.n_var):
          for hemi in range(CHOLESKY_ALL_PLOT.shape[-2]):
            
            # Plot every bootstrap's value
            # if self.plot_all_bootstraps == True:
            #   for b in range(CHOLESKY_ALL_PLOT.shape[4]):
            #     axs[row, col].plot(CHOLESKY_ALL_PLOT[i, :, row, col, b], lw = 0.5, alpha = 0.25)

            axs[row, col].plot(np.nanmedian(CHOLESKY_ALL_PLOT[i, :, row, col, hemi, :], axis = -1), label = f'Hemi {hemi}')
            axs[row, col].set_title(f'{self.var_names[row]}, {self.var_names[col]}')
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Coefficient')

            sigmas_lcl = np.nanquantile(CHOLESKY_ALL_PLOT[i, :, row, col, hemi, :], axis = -1, q = 0.16)
            sigmas_ucl = np.nanquantile(CHOLESKY_ALL_PLOT[i, :, row, col, hemi, :], axis = -1, q = 0.84)
            axs[row, col].fill_between(list(range(CHOLESKY_ALL_PLOT.shape[1])), sigmas_lcl, sigmas_ucl, alpha = 0.5)

            # Set the y-axis limits to be at the min 10% LCL and max 10% UCL
            # axs[row, col].set_ylim(
            #     np.nanmin(np.nanquantile(CHOLESKY_ALL_PLOT[i, :, row, col, :, 0], axis = -1, q = 0.1)),
            #     np.nanmax(np.nanquantile(CHOLESKY_ALL_PLOT[i, :, row, col, :, 0], axis = -1, q = 0.9))
            # )
            if row == 0 and col == 0:
              axs[row,col].legend()
      fig.suptitle(f'Experiment {i} ({self.experiment_names[i]}) Cholesky', fontsize=16)
      image_file = f"{self.image_folder_path}/cholesky_{i}{'_test' if is_test == True else ''}.png"
      plt.savefig(image_file)
      plt.close()

      print(f'Cholesky plotted at {image_file}')

  def plot_sigmas_comparison(self, exps_to_compare = [0, 1], is_test = False):

    try:
      if is_test == False:
        SIGMAS_ALL_PLOT = self.SIGMAS_ALL[:, :-self.test_size,:,:,:]
      else:
        SIGMAS_ALL_PLOT = self.SIGMAS_ALL[:, -self.test_size:,:,:,:]

      fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (6 * self.n_var, 4 * self.n_var), constrained_layout = True)
      
      for row in range(self.n_var):
        for col in range(self.n_var):
          for i in exps_to_compare:

            # If all values of precision_all_plot are nan, skip
            if np.isnan(SIGMAS_ALL_PLOT[i, :, :, :, :]).all() == True:
              continue

            axs[row, col].plot(np.nanmedian(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1), label = self.experiment_names[i])
            axs[row, col].set_title(f'{self.var_names[row]}, {self.var_names[col]}')
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Coefficient')

            sigmas_lcl = np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.16)
            sigmas_ucl = np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.84)
            axs[row, col].fill_between(list(range(SIGMAS_ALL_PLOT.shape[1])), sigmas_lcl, sigmas_ucl, alpha = 0.5)

          # Set the y-axis limits to be at the min 10% LCL and max 10% UCL
          axs[row, col].set_ylim(
              np.nanmin(np.nanquantile(SIGMAS_ALL_PLOT[exps_to_compare, :, row, col, :], axis = -1, q = 0.35)),
              np.nanmax(np.nanquantile(SIGMAS_ALL_PLOT[exps_to_compare, :, row, col, :], axis = -1, q = 0.65))
          )
          # Plot the time-invariant covariance matrix
          axs[row, col].axhline(y = np.nanmedian(self.SIGMAS_CONS_ALL[i, row, col, :]), color = 'red', label = 'Time-Invariant')

          if row == 0 and col == 0:
            axs[row, col].legend()


      fig.suptitle(f'Sigmas Comparison', fontsize=16)
      image_file = f"{self.image_folder_path}/sigmas_comparison{'_test' if is_test == True else ''}.png"
      plt.savefig(image_file)
      plt.close()
    except:
      print('Error plotting sigmas comparison')

  def plot_sigmas(self, is_test = False):
        
    # Don't show test (change this code to show in-sample)
    if is_test == False:
      SIGMAS_ALL_PLOT = self.SIGMAS_ALL[:, :-self.test_size,:,:,:]
      #cov_mat_tv_plot = cov_mat_tv[:-test_size, :, :] if self.is_simulation == True else None
    else:
      SIGMAS_ALL_PLOT = self.SIGMAS_ALL[:, -self.test_size:,:,:,:]
      #cov_mat_tv_plot = cov_mat_tv[-test_size:, :, :] if self.is_simulation == True else None

    for i in self.exps_to_plot:

      # If all values of precision_all_plot are nan, skip
      if np.isnan(SIGMAS_ALL_PLOT[i, :, :, :, :]).all() == True:
        continue

      fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (6 * self.n_var, 4 * self.n_var), constrained_layout = True)

      for row in range(self.n_var):
        for col in range(self.n_var):
          
          # Plot every bootstrap's value 
          if self.plot_all_bootstraps == True:
            for b in range(self.SIGMAS_IN_ALL.shape[4]):
              axs[row, col].plot(SIGMAS_ALL_PLOT[i, :, row, col, b], lw = 0.5, alpha = 0.25, label = i)

          axs[row, col].plot(np.nanmedian(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1))
          axs[row, col].set_title(f'{self.var_names[row]}, {self.var_names[col]}')
          axs[row, col].set_xlabel('Time')
          axs[row, col].set_ylabel('Coefficient')

          sigmas_lcl = np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.16)
          sigmas_ucl = np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.84)
          axs[row, col].fill_between(list(range(SIGMAS_ALL_PLOT.shape[1])), sigmas_lcl, sigmas_ucl, alpha = 0.5)

          #Plot the actual covariance matrix values
          # if self.is_simulation == True:
          #   axs[row, col].plot(cov_mat_tv_plot[:, row, col], color = 'black')

          # Set the y-axis limits to be at the min 10% LCL and max 10% UCL
          axs[row, col].set_ylim(
              np.nanmin(np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.35)),
              np.nanmax(np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.65))
          )
          # Plot the time-invariant covariance matrix
          axs[row, col].axhline(y = np.nanmedian(self.SIGMAS_CONS_ALL[i, row, col, :]), color = 'red', label = 'Time-Invariant')
      print('Time-Invariant Cov Mat', np.nanmedian(self.SIGMAS_CONS_ALL[i, :,:,:], axis = -1))
      print('Mean Median Time-varying Cov Mat', np.nanmean(np.nanmedian(SIGMAS_ALL_PLOT[i, :, :, :, :], axis = -1), axis = 0))

      fig.suptitle(f'Experiment {i} ({self.experiment_names[i]}) Sigma', fontsize=16)
      image_file = f"{self.image_folder_path}/sigmas_{i}{'_test' if is_test == True else ''}.png"
      plt.savefig(image_file)
      plt.close()

      # @evalmetric
      # SIGMAS_EXP: n_time, n_var, n_var, n_boot
      SIGMAS_EXP = SIGMAS_ALL_PLOT[i, :, :, :, :]
      SIGMAS_EXP_LCL = np.nanquantile(SIGMAS_EXP, axis = -1, q = 0.16)
      SIGMAS_EXP_UCL = np.nanquantile(SIGMAS_EXP, axis = -1, q = 0.84)
      SIGMAS_EXP_RANGE = SIGMAS_EXP_UCL - SIGMAS_EXP_LCL # n_time x n_var x n_var
      mean_sigmas_range = np.nanmean(SIGMAS_EXP_RANGE, axis = 0) # n_var x n_var
      high_sigmas_range = np.nanquantile(SIGMAS_EXP_RANGE, axis = 0, q = 0.95)
      
      self.evaluation_metrics.append({'metric': 'mean_sigmas_range', 'experiment': i, 'value': mean_sigmas_range})
      self.evaluation_metrics.append({'metric': '95pct_sigmas_range', 'experiment': i, 'value': high_sigmas_range})

      print(f'Cov Mat plotted at {image_file}')

  # New 9/13: Plot the correlation matrix
  def plot_corr_mat(self, is_test = False):
        
    if is_test == False:
      SIGMAS_ALL_PLOT = self.SIGMAS_ALL[:, :-self.test_size,:,:,:]
    else:
      SIGMAS_ALL_PLOT = self.SIGMAS_ALL[:, -self.test_size:,:,:,:]

    # SIGMAS_ALL_PLOT dims: n_exp, n_time, n_var, n_var, n_boot
    CORR_PLOT = np.zeros_like(SIGMAS_ALL_PLOT)
    CORR_PLOT[:] = np.nan

    # Calculate correlation matrix
    for b in range(SIGMAS_ALL_PLOT.shape[4]):
      for row in range(self.n_var):
        for col in range(self.n_var):
          CORR_PLOT[:, :, row, col, b] = SIGMAS_ALL_PLOT[:, :, row, col, b] / np.sqrt(SIGMAS_ALL_PLOT[:, :, row, row, b] * SIGMAS_ALL_PLOT[:, :, col, col, b])

    for i in self.exps_to_plot:
      fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (6 * self.n_var, 4 * self.n_var), constrained_layout = True)

      for row in range(self.n_var):
        for col in range(self.n_var):
          
          # Plot every bootstrap's value 
          if self.plot_all_bootstraps == True:
            for b in range(self.CORR_PLOT.shape[4]):
              axs[row, col].plot(CORR_PLOT[i, :, row, col, b], lw = 0.5, alpha = 0.25, label = i)

          axs[row, col].plot(np.nanmedian(CORR_PLOT[i, :, row, col, :], axis = -1))
          axs[row, col].set_title(f'{self.var_names[row]}, {self.var_names[col]}')
          axs[row, col].set_xlabel('Time')
          axs[row, col].set_ylabel('Correlation')

          corr_lcl = np.nanquantile(CORR_PLOT[i, :, row, col, :], axis = -1, q = 0.16)
          corr_ucl = np.nanquantile(CORR_PLOT[i, :, row, col, :], axis = -1, q = 0.84)
          axs[row, col].fill_between(list(range(CORR_PLOT.shape[1])), corr_lcl, corr_ucl, alpha = 0.5)

          # # Set the y-axis limits to be at the min 10% LCL and max 10% UCL
          # axs[row, col].set_ylim(
          #   -1, 1
          # )

      fig.suptitle(f'Experiment {i} ({self.experiment_names[i]}) Correlation Matrix', fontsize=16)
      image_file = f"{self.image_folder_path}/corr_mat_{i}{'_test' if is_test == True else ''}.png"
      plt.savefig(image_file)
      plt.close()

  def plot_predictions(self):

    fig, ax = plt.subplots(self.n_var, 1, figsize = (12, 3 * self.n_var), constrained_layout = True)

    for i in range(self.M_total):

      preds_median = np.nanmedian(self.PREDS_ALL[i,:,:,:], axis = 1)
      preds_test_median = np.nanmedian(self.PREDS_TEST_ALL[i,:,:,:], axis = 1)
      
      if self.is_test == False:
        preds_plot = preds_median
        actual_plot = self.Y_train
      else:
        if self.test_exclude_last == 0:
          preds_plot = preds_test_median
          actual_plot = self.Y_test
        else:
          preds_plot = preds_test_median[:-self.test_exclude_last, :]
          actual_plot = self.Y_test[:-self.test_exclude_last, :]
                
      for var in range(self.n_var):
        if i < self.M_varnn:
          ax[var].plot(preds_plot[:, var], lw = 0.75, label = self.all_names[i], color = palette[i])
        if i == self.M_total - 1:
          ax[var].plot(actual_plot[:, var], lw = 1, label = 'Actual', color = 'black')
          ax[var].set_title(self.var_names[var])
        if i >= self.M_varnn:
          ax[var].plot(preds_plot[:, var], lw = 0.75, label = self.all_names[i - self.M_varnn], ls = 'dotted')

        # Draw a vertical line at the point where we excluded data
        if self.is_test == True and self.exclude_2020 == True:
          ax[var].axvline(x = self.first_test_id_to_exclude - 0.5, ls = 'dashed', color = 'black')

        if var == 0:
          ax[var].legend()

    image_file = f'{self.image_folder_path}/preds.png'
    plt.savefig(image_file)
    plt.close()

    print(f'Predictions plotted at {image_file}')
    
  def plot_predictions_with_bands(self):
    
     ### Evaluate the predictive density for StochVol Benchmark
    
    arfit = pyreadr.read_r(f'data/stochvol_results/arfit_{self.stoch_vol_results_name}.RData')['arfit_all'].to_numpy().T
    svfit = pyreadr.read_r(f'data/stochvol_results/svfit_{self.stoch_vol_results_name}.RData')['svfit_all'].to_numpy().T
    
    preds_median_svol = np.zeros((arfit.shape[0] + 2, arfit.shape[1]))
    sigmas_svol = np.zeros((arfit.shape[0] + 2, arfit.shape[1]))
    
    if self.test_exclude_last > 0:
      preds_median_svol[2:,:] = arfit[:-self.test_exclude_last, :]
      sigmas_svol[2:,:] = svfit[:-self.test_exclude_last, :]
    else:
      preds_median_svol[2:,:] = arfit
      sigmas_svol[2:,:] = svfit
      
    # preds_median_svol and sigmas_svol dimension: (n_obs, n_var)
    preds_lcl_1sd_svol = preds_median_svol - sigmas_svol
    preds_ucl_1sd_svol = preds_median_svol + sigmas_svol   
    preds_lcl_2sd_svol = preds_median_svol - 1.96 * sigmas_svol
    preds_ucl_2sd_svol = preds_median_svol + 1.96 * sigmas_svol
    
    fig, ax = plt.subplots(self.M_varnn + 1, self.n_var, figsize = (6 * self.n_var, 4 * (self.M_varnn + 1)), constrained_layout = True)
    
    for var in range(self.n_var):
      
      ### Plot preds with bands for StochVol
      
      ax[self.M_varnn, var].plot(preds_median_svol[:, var], lw = 0.75, label = 'Median', color = 'b')
      ax[self.M_varnn, var].fill_between(list(range(preds_median_svol.shape[0])), preds_lcl_1sd_svol[:, var], preds_ucl_1sd_svol[:, var], color = 'b', alpha = 0.5)
      ax[self.M_varnn, var].fill_between(list(range(preds_median_svol.shape[0])), preds_lcl_2sd_svol[:, var], preds_ucl_2sd_svol[:, var], color = 'b', alpha = 0.25)
      ax[self.M_varnn, var].plot(self.Y_test[:, var], lw = 1, label = 'Actual', color = 'black')
      ax[self.M_varnn, var].set_title(f'{self.var_names[var]}, StochVol')
      
      # Save the y-axis limits
      y_min = ax[self.M_varnn, var].get_ylim()[0]
      y_max = ax[self.M_varnn, var].get_ylim()[1]
      
      # Plot preds with bands for VARNN
      for i in range(self.M_varnn):
        if self.is_test == False:
          preds = self.PREDS_ALL[i,:,:,:]
          actual = self.Y_train
        else:
          if self.test_exclude_last == 0:
            preds = self.PREDS_TEST_ALL[i,:,:,:]
            actual = self.Y_test
          else:
            preds = self.PREDS_TEST_ALL[i,:,:,:-self.test_exclude_last, :]
            actual = self.Y_test[:-self.test_exclude_last, :]
        
        # Calculate median and the 16th and 84th quantiles
        preds_median = np.nanmedian(preds, axis = 1)
        preds_lcl_1sd = np.nanquantile(preds, axis = 1, q = 0.16)
        preds_ucl_1sd = np.nanquantile(preds, axis = 1, q = 0.84)
        preds_lcl_2sd = np.nanquantile(preds, axis = 1, q = 0.025)
        preds_ucl_2sd = np.nanquantile(preds, axis = 1, q = 0.975)
      
         # preds: n_obs x n_bootstraps x n_var      
        # Plot the median
        ax[i, var].plot(preds_median[:, var], lw = 0.75, label = 'Median', color = 'b')
        # Plot the 16th and 84th quantiles
        ax[i, var].fill_between(list(range(preds.shape[0])), preds_lcl_1sd[:, var], preds_ucl_1sd[:, var], color = 'b', alpha = 0.5)
        # Plot the 2.5th and 97.5th quantiles
        ax[i, var].fill_between(list(range(preds.shape[0])), preds_lcl_2sd[:, var], preds_ucl_2sd[:, var], color = 'b', alpha = 0.25)
        ax[i, var].plot(actual[:, var], lw = 1, label = 'Actual', color = 'black')
        ax[i, var].set_title(f'{self.var_names[var]}, {self.all_names[i]}')
        # Set the y-axis limits to be the same as StochVol
        ax[i, var].set_ylim(y_min, y_max)
    
    
    image_file = f'{self.image_folder_path}/preds_with_bands.png'
    plt.savefig(image_file)
    plt.close()
    
    print(f'Predictions with bands plotted at {image_file}')
      

  def plot_errors(self, data_sample = 'oob', exclude_last = 0):

    '''
    Gets the median prediction from all the bootstraps models
    Calculates the Absolute Errors
    '''
        
    fig, ax = plt.subplots(1, self.n_var, figsize = (6 * self.n_var, 6), constrained_layout = True)
    for i in range(self.M_total):
      
      if data_sample == 'oob':
        preds_median = np.nanmedian(self.PREDS_ALL[i,:,:,:], axis = 1)
        error = np.abs(self.Y_train - preds_median)
      else:
        preds_median = np.nanmedian(self.PREDS_TEST_ALL[i,:,:,:], axis = 1)
        error = np.abs(self.Y_test - preds_median)
      
      if exclude_last != 0:
        error = error[:-exclude_last, :]
      
      for var in range(self.n_var):
        if i == 0:
          ax[var].set_title(self.var_names[var])
        if i >= self.M_varnn:
          ax[var].plot(error[:, var], lw = 0.5, label = self.all_names[i - self.M_varnn], ls = 'dotted')
        else:
          ax[var].plot(error[:, var], lw = 0.5, label = self.all_names[i], color = palette[i])
        if var == 0:
          ax[var].legend()
        
        # Draw a vertical line at the point where we excluded data
        if data_sample == 'test' and self.exclude_2020 == True:
          ax[var].axvline(x = self.first_test_id_to_exclude - 0.5, ls = 'dashed', color = 'black')

    plt.savefig(f'{self.image_folder_path}/error_{data_sample}.png')
    plt.close()

    fig, ax = plt.subplots(1, self.n_var, figsize = (6 * self.n_var, 6), constrained_layout = True)

    # Calculating errors
    if data_sample == 'oob':
      preds_median = np.nanmedian(self.PREDS_ALL, axis = 2)
      y_repeated = np.repeat(np.expand_dims(self.Y_train, axis = 0), self.M_total, axis = 0)
    else: # test
      preds_median = np.nanmedian(self.PREDS_TEST_ALL, axis = 2)
      y_repeated = np.repeat(np.expand_dims(self.Y_test, axis = 0), self.M_total, axis = 0)

    errors = np.abs(preds_median - y_repeated)
    if exclude_last != 0:
      errors = errors[:, :-exclude_last, :]
    
    cum_errors = np.nancumsum(errors, axis = 1)
    # dim of cum_errors: 

    # If exclude_2020, then also output the errors pre and post COVID (as well as the total errors)
    if self.exclude_2020 == True:
      errors_pre_covid = errors[:, :self.first_test_id_to_exclude, :]
      mean_absolute_errors_pre_covid = np.nanmean(errors_pre_covid, axis = 1)
      mean_absolute_errors_pre_covid_df = pd.DataFrame(mean_absolute_errors_pre_covid, columns = self.var_names, index = self.all_names)
      if self.normalize_errors_to_benchmark == True:
        mean_absolute_errors_pre_covid_df = mean_absolute_errors_pre_covid_df.div(mean_absolute_errors_pre_covid_df.iloc[self.M_varnn+3, :])
      mean_absolute_errors_pre_covid_df.to_csv(f'{self.image_folder_path}/mean_absolute_errors_pre_covid.csv')

      errors_post_covid = errors[:, self.first_test_id_to_exclude:, :]
      mean_absolute_errors_post_covid = np.nanmean(errors_post_covid, axis = 1)
      mean_absolute_errors_post_covid_df = pd.DataFrame(mean_absolute_errors_post_covid, columns = self.var_names, index = self.all_names)
      if self.normalize_errors_to_benchmark == True:
        mean_absolute_errors_post_covid_df = mean_absolute_errors_post_covid_df.div(mean_absolute_errors_post_covid_df.iloc[self.M_varnn+3, :])
      mean_absolute_errors_post_covid_df.to_csv(f'{self.image_folder_path}/mean_absolute_errors_post_covid.csv')

    mean_absolute_errors = np.nanmean(errors, axis = 1)
    mean_absolute_errors_df = pd.DataFrame(mean_absolute_errors, columns = self.var_names, index = self.all_names)
    if self.normalize_errors_to_benchmark == True: # Standardize errors by benchmark model
      mean_absolute_errors_df = mean_absolute_errors_df.div(mean_absolute_errors_df.iloc[self.M_varnn+3, :])
    mean_absolute_errors_df.T.to_csv(f'{self.image_folder_path}/mean_absolute_errors_{data_sample}.csv')

    # Choose the benchmark (fix as AR rolling)
    benchmark_cum_error = cum_errors[(self.M_varnn + 3), :, :]

    # Plots the cumulative absolute errors of the different models
    for i in range(self.M_total):
      for var in range(self.n_var):
        if i == 0:
          ax[var].set_title(self.var_names[var])
        if i >= self.M_varnn: # Make benchmarks dotted
          ax[var].plot(cum_errors[i, :, var] - benchmark_cum_error[:, var], label = self.all_names[i], color = palette[i - self.M_varnn], ls = 'dotted')
        else:
          ax[var].plot(cum_errors[i, :, var] - benchmark_cum_error[:, var], label = self.all_names[i], color = palette[i])

        if var == 0:
          ax[var].legend()
        
        # Draw a vertical line at the point where we excluded data
        if data_sample == 'test' and self.exclude_2020 == True:
          ax[var].axvline(x = self.first_test_id_to_exclude - 0.5, ls = 'dashed', color = 'black')

    image_file = f'{self.image_folder_path}/cum_errors_{data_sample}.png'
    plt.savefig(image_file)
    plt.close()

    print(f'{data_sample} Cum Errors plotted at {image_file}')

  def get_conditional_irf_eval_metrics(self):
    for exp in self.M_varnn:
      IRFS = self.Run.experiments[exp].evaluations['conditional_irf'].IRFS
      
      IRFS_MEDIAN = np.nanmedian(IRFS, axis = 1) #  n_obs x n_var x n_var x max_h
      irf_vol = np.nanstd(IRFS_MEDIAN, axis = 0) #  n_var x n_var x max_h
      self.evaluation_metrics.append({'metric': 'conditional_irf_vol', 'experiment': exp, 'value': irf_vol})


  def plot_conditional_irf_comparison_3d(self, exps_to_compare = [0, 1], is_test = False):
    print('Experiments to Compare', exps_to_compare)
    if len(exps_to_compare) > 2: # only can compare 2 experiments
      exps_to_compare = exps_to_compare[:2]
    
    for i, exp in enumerate(exps_to_compare):
      IRFS = self.Run.experiments[exp].evaluations['conditional_irf'].IRFS
      if i == 0:
        IRFS_ALL = np.zeros((len(exps_to_compare), IRFS.shape[0], IRFS.shape[1], IRFS.shape[2], IRFS.shape[3], IRFS.shape[4]))
      IRFS_ALL[i, :,:,:,:,:] = IRFS
      
    if is_test == True:
      IRFS_ALL = IRFS_ALL[:, -self.test_size:, :, :, :, :]
    else:
      IRFS_ALL = IRFS_ALL[:, :(-self.test_size), :, :, :, :]
    
    print(IRFS_ALL.shape)
    IRFS_MEDIAN_ALL = np.nanmedian(IRFS_ALL, axis = 2)

    n_exp = IRFS_MEDIAN_ALL.shape[0]
    n_var = IRFS_MEDIAN_ALL.shape[2]
    fig = make_subplots(rows = n_var, cols = n_var,
                        subplot_titles = [f'IRF {self.var_names[shock_var]} -> {self.var_names[response_var]}' for response_var in range(n_var) 
                          for shock_var in range(n_var)],
                        specs = [[{'is_3d': True} for e in range(n_var)] for e in range(n_var)],
                        shared_xaxes = False,
                        shared_yaxes = False,
                        horizontal_spacing = 0,
                        vertical_spacing = 0.05
    )

    cmap = plt.get_cmap("tab10")
    colorscale = [[0, 'rgb' + str(cmap(1)[0:3])], 
                  [1, 'rgb' + str(cmap(2)[0:3])]]

    colors = []
    colors_0 = np.zeros(shape = IRFS_MEDIAN_ALL[0, :, 0, 0, :].shape)
    colors_1 = np.ones(shape = IRFS_MEDIAN_ALL[0, :, 0, 0, :].shape)
    colors.append(colors_0)
    colors.append(colors_1)

    for shock_var in range(n_var):
      for response_var in range(n_var):
        for exp in range(n_exp):
          fig.add_trace(go.Surface(name = f'Experiment {exp}', z = IRFS_MEDIAN_ALL[exp, :, shock_var, response_var, :], 
                showscale = False, showlegend = True, 
                surfacecolor = colors[exp],
                cmin = 0, cmax = 1,
                opacity = 0.4),
                row = response_var + 1, col = shock_var + 1)

    fig.update_scenes(xaxis_title = 'Horizon',
                      yaxis_title = 'Time', 
                      zaxis_title = 'Value',
                      camera = {
                      'up': {'x': 0, 'y': 0, 'z': 1},
                      'center': {'x': 0, 'y': 0, 'z': 0},
                      'eye': {'x': 1.25, 'y': -1.5, 'z': 0.75}
                      })


    fig.update_layout(title = f'Conditional IRF Comparison', autosize=False,
                      width = 350 * n_var, height = 350 * n_var,
                      margin=dict(l=25, r=25, b=65, t=90))
    
    image_path = f"{self.image_folder_path}/irf_conditional_3d_comparison{'_test' if is_test == True else ''}.html"
    fig.write_html(image_path)


  def plot_conditional_irf_comparison(self, exps_to_compare = [0, 1]):

    try:

      print('Experiments to Compare', exps_to_compare)
      
      for i, exp in enumerate(exps_to_compare):
        IRFS = self.Run.experiments[exp].evaluations['conditional_irf'].IRFS
        if i == 0:
          IRFS_ALL = np.zeros((len(exps_to_compare), IRFS.shape[0], IRFS.shape[1], IRFS.shape[2], IRFS.shape[3], IRFS.shape[4]))
        IRFS_ALL[i, :,:,:,:,:] = IRFS
      
      print(IRFS_ALL.shape)

      IRFS_MEDIAN_ALL = np.nanmedian(IRFS_ALL, axis = 2)
      # Shape of IRFS_MEDIAN_ALL: n_models x n_obs, n_var n_var, self.max_h
      cmap = plt.cm.tab10
      fig, ax = plt.subplots(self.n_var, self.n_var, constrained_layout = True, figsize = (6 * self.n_var, 4 * self.n_var))

      alphas = [1, 0.6, 0.2]
      weights = [2, 1.5, 1]

      for shock_var in range(self.n_var):
        for response_var in range(self.n_var):
          for exp in range(len(exps_to_compare)):
            for h in [1,2,3]:
              irf_df = IRFS_MEDIAN_ALL[exp, :, shock_var, response_var, h]
              # if normalize == True:
              irf_df = irf_df / IRFS_MEDIAN_ALL[exp, :, response_var, response_var, 0] # Divide IRF by the time-0 of 
              ax[response_var, shock_var].plot(irf_df, label = f'h={h} ({self.experiment_names[exps_to_compare[exp]]})', color = cmap(exp), lw = weights[h-1], alpha = alphas[h-1])

              ax[response_var, shock_var].set_xlabel('Horizon')
              ax[response_var, shock_var].set_ylabel('Impulse Response')
              ax[response_var, shock_var].axhline(y = 0, color = 'black', ls = '--')
              ax[response_var, shock_var].set_title(f'{self.var_names[shock_var]} -> {self.var_names[response_var]}')
            if shock_var == 0 and response_var == 0:
              ax[response_var, shock_var].legend()

      fig.suptitle(f'Conditional IRF over Time Comparison', fontsize = 16)
      image_file = f'{self.image_folder_path}/irf_conditional_over_time_comparison.png'
      plt.savefig(image_file)
    except:
      print('Error plotting conditional IRF comparison')

  # Wrapper function to do all plots
  def plot_all(self, cond_irf = False):
    
    self.Run.evaluate_unconditional_irfs(self.Y_train)

    self.plot_cholesky()
    self.plot_precision()
    self.plot_sigmas()
    self.plot_corr_mat()
    self.plot_betas()

    if self.is_test == True:
      self.plot_cholesky(is_test = True)
      self.plot_precision(is_test = True)
      self.plot_sigmas(is_test = True)
      self.plot_corr_mat(is_test = True)
      self.plot_betas(is_test = True)

    self.plot_predictions()
    self.plot_errors(data_sample='oob')
    self.plot_errors(data_sample='test', exclude_last = self.test_exclude_last)
    self.evaluate_predictive_density()
    
    
    self.plot_betas_comparison(exps_to_compare = self.experiments_to_compare)
    self.plot_sigmas_comparison(exps_to_compare = self.experiments_to_compare)
  
    if self.is_test == True:
      self.plot_betas_comparison(exps_to_compare = self.experiments_to_compare, is_test = True)
      self.plot_sigmas_comparison(exps_to_compare = self.experiments_to_compare, is_test = True)

    if cond_irf == True:
      self.plot_conditional_irf_comparison_3d(exps_to_compare = self.experiments_to_compare, is_test = False)
      self.plot_conditional_irf_comparison_3d(exps_to_compare = self.experiments_to_compare, is_test = True)
      self.plot_conditional_irf_comparison(exps_to_compare = self.experiments_to_compare)
    if self.Run.execution_params['multi_forecasting'] == True:
      self.evaluate_multi_step_forecasts()

    # Save the evaluation_metrics as an npz object
    np.savez(f'{self.image_folder_path}/evaluation_metrics.npz', evaluation_metrics = self.evaluation_metrics)

  def plot_forecasts(self):
    self.plot_predictions()
    self.plot_predictions_with_bands()
    self.plot_errors(data_sample='oob')
    self.plot_errors(data_sample='test', exclude_last = self.test_exclude_last)
    self.evaluate_multi_step_forecasts()
    self.evaluate_predictive_density()

  def evaluate_multi_step_forecasts(self, exclude_last = 0):
    multi_forecasting_params = {
      'test_size': self.test_size,
      'forecast_horizons': self.Run.extensions_params['multi_forecasting']['forecast_horizons'],
      'reestimation_window': self.Run.extensions_params['multi_forecasting']['reestimation_window'],
      'benchmarks': self.Run.extensions_params['multi_forecasting']['benchmarks'],
      'n_var': self.n_var, 
      'var_names': self.var_names,
      'M_varnn': self.M_varnn,
      'exclude_last': self.Run.evaluation_params['test_exclude_last'],
      'normalize_errors_to_benchmark': self.Run.evaluation_params['normalize_errors_to_benchmark'],
      'window_length': self.Run.extensions_params['benchmarks']['window_length'],

      'dataset_name': self.dataset_name,
      'exclude_2020': self.exclude_2020,
    }
      
    ForecastMultiEvaluationObj = ForecastMultiEvaluation(self.run_name, multi_forecasting_params, 
      self.Y_train, self.Y_test)

    ForecastMultiEvaluationObj.plot_different_horizons_same_model()
    ForecastMultiEvaluationObj.plot_different_horizons()
    ForecastMultiEvaluationObj.plot_forecast_errors()
    