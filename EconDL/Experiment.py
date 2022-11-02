from datetime import datetime
import numpy as np
import os

import EconDL.DataHelpers.DataProcesser as DataProcesser
import EconDL.TrainVARNN as TrainVARNN
from EconDL.IRF.IRFConditional import IRFConditional 
from EconDL.IRF.IRFUnconditional import IRFUnconditional
from EconDL.IRF.IRFUnconditionalEvaluation import IRFUnconditionalEvaluation
from EconDL.Forecast.ForecastMulti import ForecastMulti
from EconDL.constants import keys_to_keep


class Experiment:
  def __init__(self, run_name, experiment_id, nn_hyps, run_params, execution_params, extensions_params, job_id = None, reestim_id = None):

    '''
    Notes:
    nn_hyps - contains run_params already (combined in Run)
    run_params - 
    execution_params - whether to run different extensions 
    extensions_params - params for the different extensions
    '''
    
    self.run_name = run_name
    self.experiment_id = experiment_id
    self.job_id = job_id
    self.reestim_id = reestim_id
    
    self.nn_hyps = nn_hyps
    self.run_params = run_params # run_inner_bootstraps, num_repeats, default_nn_hyps
    self.execution_params = execution_params
    self.extensions_params = extensions_params

    self.folder_path = self.run_params['folder_path']
    self.nn_hyps['num_bootstrap'] = self.run_params['num_inner_bootstraps']
    self.nn_hyps['same_train'] = self.nn_hyps['reestim_params']['same_train'] if self.nn_hyps['reestim_params']['reestim'] == True else False

    self.max_test_size = self.run_params['test_size']

    # Calculate the new test size for this instance for the reestim = True
    if self.nn_hyps['reestim_params']['reestim'] == True:
      self.reestimation_window = self.nn_hyps['reestim_params']['reestimation_window']
      self.num_reestims = self.nn_hyps['test_size'] // self.reestimation_window
      test_size = self.nn_hyps['test_size'] - (self.reestim_id if self.reestim_id else 0) * self.reestimation_window
      self.nn_hyps['test_size'] = test_size
      self.nn_hyps['max_test_size'] = self.max_test_size
    else:
      self.num_reestims = 1
      self.reestimation_window = self.nn_hyps['test_size']
      self.nn_hyps['max_test_size'] = self.max_test_size

    self.results_uncompiled = []
    self.results = []
    self.is_trained = False
    self.is_compiled = False

    self.evaluations = {
      'conditional_irf': None,
      'unconditional_irf': None,
      'multi_forecasting': None
    }

    print(f'Experiment Initialized: experiment {self.experiment_id}, repeat {self.job_id}, reestim {self.reestim_id}, num reestims {self.num_reestims}')
    self.load_results()

  def check_results_sizes(self):
    for k in keys_to_keep.keys():
      print(k, self.results[k].shape)

  # Only done after all repeats are trained (called by Evaluation function)
  def compute_conditional_irfs(self):
    if self.execution_params['conditional_irfs'] == False or self.job_id is not None:
      print('Experiment compute_conditional_irfs(): Conditional IRFs turned off')
      return
    if os.path.exists(f'{self.folder_path}/images/irf_conditional_{self.experiment_id}.png'):
      print('Experiment compute_conditional_irfs(): Already plotted Conditional IRFs')
      return 

    if self.is_trained == True:
      image_folder_path = self.run_params['image_folder_path']
      image_file = f'{image_folder_path}/irf_conditional_{self.experiment_id}.png'

      conditional_irf_params = {
        'n_var': self.nn_hyps['n_var'],
        'var_names': self.nn_hyps['var_names'],
        'n_lags': self.nn_hyps['n_lag_linear'],
        'n_betas': self.nn_hyps['n_var'] * self.nn_hyps['n_lag_linear'] + 1,
        'max_h': self.extensions_params['conditional_irfs']['max_h'],
        'test_exclude_last': self.extensions_params['conditional_irfs']['test_exclude_last'],
        'test_size': self.max_test_size,
        'dataset': self.run_params['dataset'],
        'experiment_id': self.experiment_id,
        'experiment_name': self.nn_hyps['name']
      }
      #try:
      IRFConditionalObj = IRFConditional(self.results, conditional_irf_params)
      IRFConditionalObj.plot_irfs(image_folder_path)
      IRFConditionalObj.plot_irfs_3d(image_folder_path, is_test = False)
      IRFConditionalObj.plot_irfs_3d(image_folder_path, is_test = True)
      IRFConditionalObj.plot_irfs_over_time(image_folder_path, normalize = self.extensions_params['conditional_irfs']['normalize_time_plot'])
      self.evaluations['conditional_irf'] = IRFConditionalObj

  def compute_unconditional_irfs(self, Y_train, Y_test, X_train, results, device):
    if self.execution_params['unconditional_irfs'] == False:
      print('Experiment compute_unconditional_irfs(): Unconditional IRFs turned off')
      return 
    # results contains the trained model
    unconditional_irf_params = {
        'n_lag_linear': self.nn_hyps['n_lag_linear'],
        'n_lag_d': self.nn_hyps['n_lag_d'],
        'n_var': len(self.nn_hyps['var_names']),
        'num_simulations': self.extensions_params['unconditional_irfs']['num_simulations'],
        'endh': self.extensions_params['unconditional_irfs']['endh'],
        'start_shock_time': self.extensions_params['unconditional_irfs']['start_shock_time'],
        'forecast_method': self.extensions_params['unconditional_irfs']['forecast_method'], # old or new
        'max_h': self.extensions_params['unconditional_irfs']['max_h'], 
        'var_names': self.nn_hyps['var_names'],
        'end_precision_lambda': self.nn_hyps['end_precision_lambda'],
        'model': 'VARNN'
      }

    # Check if multiple hemispheres, or FCN
    if ('s_pos_setting' in self.nn_hyps.keys() and self.nn_hyps['s_pos_setting']['hemis'] in ['combined', 'time', 'endog_time', 'endog_exog', 'exog', 'endog_exog_time']) or self.nn_hyps['fcn'] == True:
      print('Experiment compute_unconditional_irfs(): Experiment has multiple hemispheres / exogenous data / FCN, not training unconditional IRFs')
      fcast = np.zeros((self.extensions_params['unconditional_irfs']['num_simulations'], len(self.nn_hyps['var_names']), len(self.nn_hyps['var_names']), 3))
      fcast[:] = np.nan
      with open(f'{self.folder_path}/fcast_params_{self.experiment_id}_repeat_{self.job_id}_reestim_{self.reestim_id}.npz', 'wb') as f:
        np.savez(f, fcast = fcast, fcast_cov_mat = None)

    else:

      IRFUnconditionalObj = IRFUnconditional(unconditional_irf_params, device)
      fcast, fcast_cov_mat, sim_shocks = IRFUnconditionalObj.get_irfs_wrapper(Y_train, Y_test, X_train, results)

      with open(f'{self.folder_path}/fcast_params_{self.experiment_id}_repeat_{self.job_id}_reestim_{self.reestim_id}.npz', 'wb') as f:
        np.savez(f, fcast = fcast, fcast_cov_mat = fcast_cov_mat)

  def compute_multi_forecasts(self, X_train, X_test, Y_train, Y_test, results, nn_hyps, device):

    if self.execution_params['multi_forecasting'] == False:
      print('Experiment compute_multi_forecasts(): Multi Forecasting turned off')
      return 

    multi_forecasting_params = {
      'test_size': self.nn_hyps['test_size'], 
      'forecast_horizons': self.extensions_params['multi_forecasting']['forecast_horizons'],
      'reestimation_window': self.reestimation_window,
      'num_inner_bootstraps': self.nn_hyps['num_inner_bootstraps'],
      'num_sim_bootstraps': self.extensions_params['multi_forecasting']['num_sim_bootstraps'],
      'num_repeats': 1, 

      'n_lag_linear': self.nn_hyps['n_lag_linear'],
      'n_lag_d': self.nn_hyps['n_lag_d'],
      'n_var': len(self.nn_hyps['var_names']),
      'forecast_method': self.extensions_params['multi_forecasting']['forecast_method'] if self.nn_hyps['fcn'] == False else 'old', # old or new
      'var_names': self.nn_hyps['var_names'],
      'end_precision_lambda': self.nn_hyps['end_precision_lambda'],
      'model': 'VARNN'
    }

    ForecastMultiObj = ForecastMulti(self.run_name, Y_train, Y_test, multi_forecasting_params, device = device)
    
    # Benchmarks
    FCAST = ForecastMultiObj.conduct_multi_forecasting_wrapper(X_train, X_test, results, nn_hyps)

    print('Experiment compute_multi_forecasts(): Done with Multiforecasting')
    with open(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_repeat_{self.job_id}_reestim_{self.reestim_id}.npz', 'wb') as f:
      np.savez(f, fcast = FCAST)

  # @DEV: Don't pass in the dataset in the _init_() because if not then there will be multiple copies of
  # the dataset sitting in each run.
  # Results are now only compiled during evaluation
  def train(self, dataset, exog_dataset, device):

    if self.is_trained == True:
      print('Experiment train(): Trained already')
    else:
      if self.nn_hyps['exog_data'] == True and exog_dataset is not None and self.nn_hyps['exog'] is None:
        self.nn_hyps['exog'] = np.array(exog_dataset)

      X_train, X_test, Y_train, Y_test, nn_hyps = DataProcesser.process_data_wrapper(dataset, self.nn_hyps)
      
      repeat_ids = []
      if self.job_id is not None:
        repeat_ids = [self.job_id]
      else:
        repeat_ids = range(self.run_params['num_repeats'])
      # For each repeat
      for repeat_id in repeat_ids:

        results = TrainVARNN.conduct_bootstrap(X_train, X_test, Y_train, Y_test, nn_hyps, device)
        
        folder_path = self.run_params['folder_path']

        results_saved = {
            'betas': results['betas_draws'], 
            'betas_in': results['betas_in_draws'], 
            'sigmas': results['sigmas_draws'], 
            'sigmas_in': results['sigmas_in_draws'],
            'precision': results['precision_draws'], 
            'precision_in': results['precision_in_draws'],
            'cholesky': results['cholesky_draws'], 
            'cholesky_in': results['cholesky_in_draws'],
            'train_preds': results['pred_in_ensemble'] , 
            'test_preds': results['pred_ensemble'], 
            'y': Y_train, 
            'y_test': Y_test, 
            'params': nn_hyps
        }

        with open(f'{folder_path}/params_{self.experiment_id}_repeat_{repeat_id}_reestim_{self.reestim_id}.npz', 'wb') as f:
          np.savez(f, results = results_saved)

        self.results_uncompiled.append(results_saved)
        print(f'Experiment train(): Finished training repeat {repeat_id} of experiment {self.experiment_id} at {datetime.now()}')

        self.compute_unconditional_irfs(Y_train, Y_test, X_train, results, device = device)
        self.compute_multi_forecasts(X_train, X_test, Y_train, Y_test, results, nn_hyps, device)

      # After completing all repeats
      self.is_trained = True

  # Used at Evaluation time (called by evaluation object)
  # If repeats_to_include is None, then call all
  def compile_all(self, repeats_to_include = None):
    self._compile_results()
    self._compile_reestims()
    self._compile_multi_forecasting_results(repeats_to_include)
    self._compile_multi_forecasting_reestims()
    self._compile_unconditional_irf_results(repeats_to_include)

  def _compile_reestims(self):
    # Assume that the repeats are already compiled
    
    ''' 
    Compiling the Reestimations:
    Latest combine all the TEST betas, sigmas, cholesky, precision, preds
    '''
    # Return if the reestims are already compiled
    if self.is_compiled == True:
      print('Experiment _compile_reestims(): Compiled results already')
      return

    all_size = self.results[0]['betas'].shape[0] # This will be the size of the entire train+test
    orig_train_size = all_size - self.max_test_size # Train size for first reestim is the size of the training set because we start from largest test set

    for reestim in range(self.num_reestims):
      reestim_results = self.results[reestim]
      
      BETAS_ALL = reestim_results['betas']
      SIGMAS_ALL = reestim_results['sigmas']
      PRECISION_ALL = reestim_results['precision']
      CHOLESKY_ALL = reestim_results['cholesky']
      PREDS_TEST = reestim_results['test_preds']
      print('BETAS', BETAS_ALL.shape, 'SIGMAS', SIGMAS_ALL.shape, 'PRECISION', PRECISION_ALL.shape, 'CHOLESKY', CHOLESKY_ALL.shape, 'PREDS_TEST', PREDS_TEST.shape)
      
      train_size_this_reestim = all_size - PREDS_TEST.shape[0]
      test_size_this_reestim = PREDS_TEST.shape[0]
      print(f'Reestim {reestim}: Train size {train_size_this_reestim}, Test size {test_size_this_reestim}')
      
      # Break into train and test
      # Train has same length for all reestims
      
      # Test has different length for each reestim (getting shorter and shorter)
      BETAS_TEST = BETAS_ALL[(-test_size_this_reestim):, :, :, :, :]
      SIGMAS_TEST = SIGMAS_ALL[(-test_size_this_reestim):, :, :, :]
      PRECISION_TEST = PRECISION_ALL[(-test_size_this_reestim):, :, :, :]
      CHOLESKY_TEST = CHOLESKY_ALL[(-test_size_this_reestim):, :, :, :, :]

      if reestim == 0:
        # Ensures that we get the 1st reestim's BETAS_TRAIN
        BETAS_TRAIN = BETAS_ALL[:orig_train_size, :, :, :, :]
        SIGMAS_TRAIN = SIGMAS_ALL[:orig_train_size, :, :, :]
        PRECISION_TRAIN = PRECISION_ALL[:orig_train_size, :, :, :]
        CHOLESKY_TRAIN = CHOLESKY_ALL[:orig_train_size, :, :, :, :]
        
        # Create TEST_ALL which is the size of max_test_size
        BETAS_TEST_ALL = np.zeros((self.max_test_size, BETAS_TEST.shape[1], BETAS_TEST.shape[2], BETAS_TEST.shape[3], BETAS_TEST.shape[4], self.num_reestims))
        SIGMAS_TEST_ALL = np.zeros((self.max_test_size, SIGMAS_TEST.shape[1], SIGMAS_TEST.shape[2], SIGMAS_TEST.shape[3], self.num_reestims))
        PRECISION_TEST_ALL = np.zeros((self.max_test_size, PRECISION_TEST.shape[1], PRECISION_TEST.shape[2], PRECISION_TEST.shape[3], self.num_reestims))
        CHOLESKY_TEST_ALL = np.zeros((self.max_test_size, CHOLESKY_TEST.shape[1], CHOLESKY_TEST.shape[2], CHOLESKY_TEST.shape[3], CHOLESKY_TEST.shape[4], self.num_reestims))
        PREDS_TEST_ALL = np.zeros((self.max_test_size, PREDS_TEST.shape[1], PREDS_TEST.shape[2], self.num_reestims))
        
        BETAS_TEST_ALL[:] = np.nan
        SIGMAS_TEST_ALL[:] = np.nan
        PRECISION_TEST_ALL[:] = np.nan
        CHOLESKY_TEST_ALL[:] = np.nan
        PREDS_TEST_ALL[:] = np.nan
        
      # Fill in the results for this reestim
      BETAS_TEST_ALL[-test_size_this_reestim:, :, :, :, :, reestim] = BETAS_TEST
      SIGMAS_TEST_ALL[-test_size_this_reestim:, :, :, :, reestim] = SIGMAS_TEST
      PRECISION_TEST_ALL[-test_size_this_reestim:, :, :, :, reestim] = PRECISION_TEST
      CHOLESKY_TEST_ALL[-test_size_this_reestim:, :, :, :, :, reestim] = CHOLESKY_TEST
      PREDS_TEST_ALL[-test_size_this_reestim:, :, :, reestim] = PREDS_TEST

    print('BETAS_TEST_ALL', BETAS_TEST_ALL.shape, 'SIGMAS_TEST_ALL', SIGMAS_TEST_ALL.shape, 'PRECISION_TEST_ALL', PRECISION_TEST_ALL.shape, 'CHOLESKY_TEST_ALL', CHOLESKY_TEST_ALL.shape, 'TEST_PREDS_ALL', PREDS_TEST_ALL.shape)
    
    # TEST_LATEST which stores the results of the latest reestim for the test set
    BETAS_TEST_LATEST = np.zeros_like(BETAS_TEST_ALL)[:,:,:,:,:,-1]
    SIGMAS_TEST_LATEST = np.zeros_like(SIGMAS_TEST_ALL)[:,:,:,:,-1]
    PRECISION_TEST_LATEST = np.zeros_like(PRECISION_TEST_ALL)[:,:,:,:,-1]
    CHOLESKY_TEST_LATEST = np.zeros_like(CHOLESKY_TEST_ALL)[:,:,:,:,:,-1]
    PREDS_TEST_LATEST = np.zeros_like(PREDS_TEST_ALL)[:,:,:,-1]
    
    # TRAIN_LATEST which stores the results of the latest reestim for the train set
    BETAS_TEST_LATEST[:] = np.nan
    SIGMAS_TEST_LATEST[:] = np.nan
    PRECISION_TEST_LATEST[:] = np.nan
    CHOLESKY_TEST_LATEST[:] = np.nan
    PREDS_TEST_LATEST[:] = np.nan

    for reestim in range(self.num_reestims): # for num_reestims
      # Assign the test betas according to the reestimation results
      BETAS_TEST_LATEST[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, :, :] = BETAS_TEST_ALL[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, :, :, reestim]
      SIGMAS_TEST_LATEST[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, :] = SIGMAS_TEST_ALL[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, :, reestim]
      PRECISION_TEST_LATEST[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, :] = PRECISION_TEST_ALL[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, :, reestim]
      CHOLESKY_TEST_LATEST[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, :, :] = CHOLESKY_TEST_ALL[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, :, :, reestim]
      PREDS_TEST_LATEST[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :] = PREDS_TEST_ALL[(reestim * self.reestimation_window):((reestim + 1) * self.reestimation_window), :, :, reestim]
      
    # Combine TRAIN and TEST to get output
    BETAS_OUT = np.concatenate((BETAS_TRAIN, BETAS_TEST_LATEST), axis=0)
    SIGMAS_OUT = np.concatenate((SIGMAS_TRAIN, SIGMAS_TEST_LATEST), axis=0)
    PRECISION_OUT = np.concatenate((PRECISION_TRAIN, PRECISION_TEST_LATEST), axis=0)
    CHOLESKY_OUT = np.concatenate((CHOLESKY_TRAIN, CHOLESKY_TEST_LATEST), axis=0)

    print('Compiled Re-estimations', 'BETAS_OUT', BETAS_OUT.shape, 'SIGMAS_OUT', SIGMAS_OUT.shape, 'PRECISION_OUT', PRECISION_OUT.shape, 'CHOLESKY_OUT', CHOLESKY_OUT.shape, 'TEST_OUT', PREDS_TEST_ALL.shape)

    reestim_results = self.results[0] # Load the first reestimation results
    results_reestim_compiled = {
      'betas': BETAS_OUT,
      'betas_in': reestim_results['betas_in'],
      'sigmas': SIGMAS_OUT,
      'sigmas_in': reestim_results['sigmas_in'],
      'precision': PRECISION_OUT,
      'precision_in': reestim_results['precision_in'],
      'cholesky': CHOLESKY_OUT,
      'cholesky_in': reestim_results['cholesky_in'],
      'train_preds': reestim_results['train_preds'],
      'test_preds': PREDS_TEST_LATEST,
      'y': reestim_results['y'],
      'y_test': reestim_results['y_test'],
      'params': reestim_results['params']
    }

    self.results = results_reestim_compiled

    with open(f'{self.folder_path}/params_{self.experiment_id}_compiled.npz', 'wb') as f:
      np.savez(f, results = results_reestim_compiled)


  # Compile results if there are multiple repeats (in results)slack
  def _compile_results(self):
    
    if self.job_id is not None:
      print('Experiment _compile_results(): Multiple Jobs, compiling turned off')
      return
    if self.is_compiled == True:
      print('Experiment _compile_results(): Compiled results already')
      return
    
    #print('results_uncompiled num_reestims', len(self.results_uncompiled))
    #print('number of repeats', len(self.results_uncompiled[0]))

    for reestim in range(self.num_reestims):

      reestim_results_uncompiled = self.results_uncompiled[reestim]

      repeat_id = 0
      # While there are more repeats to process, stack the fields
      while repeat_id < len(reestim_results_uncompiled):
        results_repeat = reestim_results_uncompiled[repeat_id]

        if repeat_id == 0:
          results_compiled = {k:v for k,v in results_repeat.items() if k in keys_to_keep.keys()}
        else:
          for k, v in results_compiled.items():
            results_compiled[k] = np.concatenate([results_compiled[k], results_repeat[k]], axis = keys_to_keep[k])
        repeat_id += 1

      results_compiled['y'] = results_repeat['y']
      results_compiled['y_test'] = results_repeat['y_test']
      results_compiled['params'] = results_repeat['params']

      # Assign the compiled results back to the original results
      self.results.append(results_compiled)
      with open(f'{self.folder_path}/params_{self.experiment_id}_reestim_{reestim}_compiled.npz', 'wb') as f:
        np.savez(f, results = results_compiled)
    
    # self.is_compiled = True

  def _compile_multi_forecasting_reestims(self):

    for reestim in range(self.num_reestims):
      FCAST = np.load(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_reestim_{reestim}_compiled.npz')['fcast']

      #print('FCAST', FCAST.shape)
      if reestim == 0:
        FCAST_ALL = FCAST
      else:
        FCAST_TEMP = np.zeros_like(FCAST_ALL)[:,:,:,:,0] # Create this so that TEST_PREDS.shape[0] is max_test_size instead of the test_size for that reestim
        FCAST_TEMP[:, :, :, (reestim * self.reestimation_window):] = FCAST[:,:,:,:,0]
        FCAST_ALL = np.concatenate((FCAST_ALL, np.expand_dims(FCAST_TEMP, axis = -1)), axis = -1)
    
    print('FCAST_ALL', FCAST_ALL.shape)
    FCAST_OUT = np.zeros_like(FCAST_ALL)[:,:,:,:,-1]
    for reestim in range(self.num_reestims):
      FCAST_OUT[:,:,:,(reestim * self.reestimation_window):] = FCAST_ALL[:,:,:,(reestim * self.reestimation_window):, reestim]
    FCAST_OUT = np.expand_dims(FCAST_OUT, axis = -1)

    print('Compiled Multi-Forecasting Re-estimations', 'FCAST_OUT', FCAST_OUT.shape)
    with open(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_compiled.npz', 'wb') as f:
      np.savez(f, fcast = FCAST_OUT)


  def _compile_multi_forecasting_results(self, repeats_to_include = None):

    if self.execution_params['multi_forecasting'] == False:
      print('Experiment _compile_multi_forecasting_results(): Multi Forecasting turned off')
      return 
    if self.job_id is not None:
      print('Experiment _compile_multi_forecasting_results(): Multiple Jobs, compiling turned off')
      return

    if repeats_to_include is None:

      for reestim_id in range(self.num_reestims):
          
        num_repeats = self.run_params['num_repeats']
        if os.path.exists(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_repeat_0_reestim_{reestim_id}.npz') == False:
          print('Experiment _compile_multi_forecasting_results(): No Multi-forecasting results')
          return

        repeat_id = 0
        while repeat_id < num_repeats:

          fcast_repeat = np.load(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_repeat_{repeat_id}_reestim_{reestim_id}.npz', allow_pickle=True)['fcast']
          if repeat_id == 0:
            fcast_all = fcast_repeat # horizons x variables x bootstraps x time x reestimation_window
          else:
            fcast_all = np.concatenate([fcast_all, fcast_repeat], axis = 2)
          repeat_id += 1
        
        with open(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_reestim_{reestim_id}_compiled.npz', 'wb') as f:
          np.savez(f, fcast = fcast_all)
    
    else:
      if os.path.exists(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_repeat_{repeats_to_include[0]}.npz') == False:
        print('Experiment _compile_multi_forecasting_results(): No Multi-forecasting results')
        return

      for reestim_id in range(self.num_reestims):
        for i, repeat_id in enumerate(repeats_to_include):
          fcast_repeat = np.load(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_repeat_{repeat_id}.npz', allow_pickle=True)['fcast']
          if i == 0:
            fcast_all = fcast_repeat # horizons x variables x bootstraps x time x reestimation_window
          else:
            fcast_all = np.concatenate([fcast_all, fcast_repeat], axis = 2)
      
      with open(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_reestim_{reestim_id}_compiled.npz', 'wb') as f:
          np.savez(f, fcast = fcast_all)
            
    

  def _compile_unconditional_irf_results(self, repeats_to_include = None):

    if self.job_id is not None:
      print('Experiment _compile_unconditional_irf_results(): Multiple Jobs, compiling turned off')
      return
    if self.execution_params['unconditional_irfs'] == False:
      print('Experiment _compile_unconditional_irf_results(): Unconditional IRFs turned off')
      return 
    if ('s_pos_setting' in self.nn_hyps.keys() and self.nn_hyps['s_pos_setting']['hemis'] in ['combined', 'time', 'endog_time', 'endog_exog', 'exog', 'endog_exog_time']):
      print('Experiment _compile_unconditional_irf_results(): Experiment has multiple hemispheres, not training unconditional IRFs')
      return
      
    if repeats_to_include is None:
      
      num_repeats = self.run_params['num_repeats']
      repeat_id = 0
      while repeat_id < num_repeats:

        out_repeat = np.load(f'{self.folder_path}/fcast_params_{self.experiment_id}_repeat_{repeat_id}_reestim_0.npz', allow_pickle=True)
        fcast = out_repeat['fcast']
        #fcast_cov_mat = out_repeat['fcast_cov_mat']
        if repeat_id == 0:
          fcast_all = np.zeros((num_repeats, fcast.shape[0], fcast.shape[1], fcast.shape[2], fcast.shape[3]))
          #fcast_cov_mat_all = np.zeros((num_repeats, fcast_cov_mat.shape[0], fcast_cov_mat.shape[1], fcast_cov_mat.shape[2], fcast_cov_mat.shape[3], fcast_cov_mat.shape[4]))
          fcast_cov_mat_all = None
        fcast_all[repeat_id, :,:,:,:] = fcast
        #fcast_cov_mat_all[repeat_id, :,:,:,:,:] = fcast_cov_mat
        repeat_id += 1
    else:
      for i, repeat_id in enumerate(repeats_to_include):
        out_repeat = np.load(f'{self.folder_path}/fcast_params_{self.experiment_id}_repeat_{repeat_id}_reestim_0.npz', allow_pickle=True)
        fcast = out_repeat['fcast']
        if i == 0:
          fcast_all = np.zeros((len(repeats_to_include), fcast.shape[0], fcast.shape[1], fcast.shape[2], fcast.shape[3]))
          #fcast_cov_mat_all = np.zeros((num_repeats, fcast_cov_mat.shape[0], fcast_cov_mat.shape[1], fcast_cov_mat.shape[2], fcast_cov_mat.shape[3], fcast_cov_mat.shape[4]))
          fcast_cov_mat_all = None
        fcast_all[i, :,:,:,:] = fcast

    
    with open(f'{self.folder_path}/fcast_params_{self.experiment_id}_compiled.npz', 'wb') as f:
      np.savez(f, fcast = fcast_all)


  def evaluate_unconditional_irf_results(self, Y_train):

    print('s_pos', self.nn_hyps['s_pos_setting'])

    if ('s_pos_setting' in self.nn_hyps.keys() and self.nn_hyps['s_pos_setting']['hemis'] in ['combined', 'time', 'endog_time', 'endog_exog', 'exog', 'endog_exog_time']):
      print('Experiment _compile_unconditional_irf_results(): Experiment has multiple hemispheres, no unconditional IRF results to compile')
      return

    results = np.load(f'{self.folder_path}/fcast_params_{self.experiment_id}_compiled.npz', allow_pickle = True)
    fcast_all = results['fcast']
    fcast_cov_mat_all = None

    unconditional_irf_params = {
        'n_lag_linear': self.nn_hyps['n_lag_linear'],
        'n_lag_d': self.nn_hyps['n_lag_d'],
        'n_var': len(self.nn_hyps['var_names']),
        'num_simulations': self.extensions_params['unconditional_irfs']['num_simulations'],
        'endh': self.extensions_params['unconditional_irfs']['endh'],
        'start_shock_time': self.extensions_params['unconditional_irfs']['start_shock_time'],
        'forecast_method': self.extensions_params['unconditional_irfs']['forecast_method'], # old or new
        'max_h': self.extensions_params['unconditional_irfs']['max_h'], 
        'var_names': self.nn_hyps['var_names'],
        'plot_all_bootstraps': self.extensions_params['unconditional_irfs']['plot_all_bootstraps'],
        'experiment_id': self.experiment_id,
        'experiment_name': self.nn_hyps['name']
      }

    IRFUnconditionalEvaluationObj = IRFUnconditionalEvaluation(fcast_all, fcast_cov_mat_all, unconditional_irf_params, Y_train)
    IRFUnconditionalEvaluationObj.evaluate_unconditional_irfs()
    IRFUnconditionalEvaluationObj.plot_irfs(self.run_params['image_folder_path'], self.experiment_id)
    IRFUnconditionalEvaluationObj.plot_cumulative_irfs(self.run_params['image_folder_path'], self.experiment_id)

    self.evaluations['unconditional_irf'] = IRFUnconditionalEvaluationObj
    
  def load_results(self, repeats_to_include = None):

    print(f'Experiment load_results(): Loading results for Experiment id {self.experiment_id}, reestims: {self.num_reestims}, repeats_to_include: {repeats_to_include}')
    # Check if the compiled results exist
    if os.path.exists(f'{self.folder_path}/params_{self.experiment_id}_compiled.npz'):
      self.is_trained = True
      load_file = f'{self.folder_path}/params_{self.experiment_id}_compiled.npz'
      results_loaded = np.load(load_file, allow_pickle = True)['results'].item()
      self.results = results_loaded
      self.is_compiled = True
      print(f'Experiment load_results(): Loaded compiled results')
      return

    # Check if results for this repeat exist

    # If all repeats are to be included
    elif repeats_to_include == None and os.path.exists(f'{self.folder_path}/params_{self.experiment_id}_repeat_{self.job_id if self.job_id else 0}_reestim_{self.reestim_id if self.reestim_id else 0}.npz'):

      # self.results_uncompiled is List of lists, each inner list is one reestim, each element is one repeat in that reestim 
      self.results_uncompiled = []
      for reestim_id in range(self.num_reestims):
        
        reestim_results_uncompiled = []
        repeat_id = 0
        
        while os.path.exists(f'{self.folder_path}/params_{self.experiment_id}_repeat_{repeat_id}_reestim_{reestim_id}.npz'):
          self.is_trained = True
          load_file = f'{self.folder_path}/params_{self.experiment_id}_repeat_{repeat_id}_reestim_{reestim_id}.npz'
          results_loaded = np.load(load_file, allow_pickle = True)['results'].item()
          reestim_results_uncompiled.append(results_loaded)

          print(f'Experiment load_results(): Loaded results for reestim {reestim_id}, repeat {repeat_id}')
          repeat_id += 1
        
        self.results_uncompiled.append(reestim_results_uncompiled)

      return
    
    # If some repeats are to be skipped - deprecated
    elif repeats_to_include is not None and os.path.exists(f'{self.folder_path}/params_{self.experiment_id}_repeat_{repeats_to_include[0]}_reestim_{self.reestim_id if self.reestim_id else 0}.npz'):

      self.results_uncompiled = []
      self.is_trained = True
      
      for reestim_id in range(self.num_reestims):
        reestim_results_uncompiled = []
      
        for repeat_id in repeats_to_include:
          load_file = f'{self.folder_path}/params_{self.experiment_id}_repeat_{repeat_id}.npz'
          results_loaded = np.load(load_file, allow_pickle = True)['results'].item()
          reestim_results_uncompiled.append(results_loaded)

          print(f'Experiment load_results(): Loaded results for reestim {reestim_id}, repeat {repeat_id}')
        
        self.results_uncompiled.append(reestim_results_uncompiled)

    else:
      print('Experiment load_results(): Not trained yet')

  def __str__(self):
    return f'''
    Run Name: {self.run_name}
    Experiment ID: {self.experiment_id}
    nn_hyps: {self.nn_hyps}
    is_trained: {self.is_trained}
    '''


  
