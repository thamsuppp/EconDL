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
  def __init__(self, run_name, experiment_id, nn_hyps, run_params, execution_params, extensions_params, job_id = None):

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
    
    self.nn_hyps = nn_hyps
    self.run_params = run_params # run_inner_bootstraps, num_repeats, default_nn_hyps
    self.execution_params = execution_params
    self.extensions_params = extensions_params

    self.folder_path = self.run_params['folder_path']
    self.nn_hyps['num_bootstrap'] = self.run_params['num_inner_bootstraps']

    self.results_uncompiled = []
    self.results = None
    self.is_trained = False
    self.is_compiled = False

    self.evaluations = {
      'conditional_irf': None,
      'unconditional_irf': None,
      'multi_forecasting': None
    }

    self.load_results()

  def check_results_sizes(self):
    for k in keys_to_keep.keys():
      print(k, self.results[k].shape)

  def compute_conditional_irfs(self):
    if self.execution_params['conditional_irfs'] == False:
      print('Unconditional IRFs turned off')
      return 

    if self.is_trained == True:
      image_folder_path = self.run_params['image_folder_path']
      image_file = f'{image_folder_path}/irf_conditional_{self.experiment_id}.png'

      conditional_irf_params = {
        'n_var': self.nn_hyps['n_var'],
        'var_names': self.nn_hyps['var_names'],
        'n_lags': self.nn_hyps['n_lag_linear'],
        'n_betas': self.nn_hyps['n_var'] * self.nn_hyps['n_lag_linear'] + 1,
        'max_h': self.extensions_params['conditional_irfs']['max_h']
      }
      
      IRFConditionalObj = IRFConditional(self.results, conditional_irf_params)
      IRFConditionalObj.plot_irfs(image_folder_path, self.experiment_id)
      self.evaluations['conditional_irf'] = IRFConditionalObj

  def compute_unconditional_irfs(self, Y_train, Y_test, results, device, repeat_id):
    if self.execution_params['unconditional_irfs'] == False:
      print('Unconditional IRFs turned off')
      return 
    # results contains the trained model
    unconditional_irf_params = {
        'n_lag_linear': self.nn_hyps['n_lag_linear'],
        'n_lag_d': self.nn_hyps['n_lag_d'],
        'n_var': len(self.nn_hyps['variables']),
        'num_simulations': self.extensions_params['unconditional_irfs']['num_simulations'],
        'endh': self.extensions_params['unconditional_irfs']['endh'],
        'start_shock_time': self.extensions_params['unconditional_irfs']['start_shock_time'],
        'forecast_method': self.extensions_params['unconditional_irfs']['forecast_method'], # old or new
        'max_h': self.extensions_params['unconditional_irfs']['max_h'], 
        'var_names': self.nn_hyps['variables'],
        'model': 'VARNN'
      }

    IRFUnconditionalObj = IRFUnconditional(unconditional_irf_params, device)
    fcast, fcast_cov_mat, sim_shocks = IRFUnconditionalObj.get_irfs_wrapper(Y_train, Y_test, results)

    with open(f'{self.folder_path}/fcast_params_{self.experiment_id}_repeat_{repeat_id}.npz', 'wb') as f:
      np.savez(f, fcast = fcast, fcast_cov_mat = fcast_cov_mat)

  def compute_multi_forecasts(self, X_train, X_test, Y_train, Y_test, results, nn_hyps, device, repeat_id):

    if self.execution_params['multi_forecasting'] == False:
      print('Multi Forecasting turned off')
      return 

    multi_forecasting_params = {
      'test_size': self.nn_hyps['test_size'], 
      'forecast_horizons': self.extensions_params['multi_forecasting']['forecast_horizons'],
      'reestimation_window': self.extensions_params['multi_forecasting']['reestimation_window'],
      'num_inner_bootstraps': self.nn_hyps['num_inner_bootstraps'],
      'num_sim_bootstraps': self.extensions_params['multi_forecasting']['num_sim_bootstraps'],
      'num_repeats': 1, 

      'n_lag_linear': self.nn_hyps['n_lag_linear'],
      'n_lag_d': self.nn_hyps['n_lag_d'],
      'n_var': len(self.nn_hyps['variables']),
      'forecast_method': self.extensions_params['multi_forecasting']['forecast_method'], # old or new
      'var_names': self.nn_hyps['variables'],
      'model': 'VARNN'
    }

    ForecastMultiObj = ForecastMulti(self.run_name, Y_train, Y_test, multi_forecasting_params, device = device)
    
    # Benchmarks
    FCAST = ForecastMultiObj.conduct_multi_forecasting_wrapper(X_train, X_test, results, nn_hyps)

    print('Done with Multiforecasting')
    with open(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_repeat_{repeat_id}.npz', 'wb') as f:
      np.savez(f, fcast = FCAST)


  # @DEV: Don't pass in the dataset in the _init_() because if not then there will be multiple copies of
  # the dataset sitting in each run.
  def train(self, dataset, device):

    if self.is_trained == True:
      print('Trained already')
    else:
      X_train, X_test, Y_train, Y_test, nn_hyps = DataProcesser.process_data_wrapper(dataset, self.nn_hyps)
      
      repeat_ids = []
      if self.job_id is not None:
        repeat_ids = self.job_id
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

        with open(f'{folder_path}/params_{self.experiment_id}_repeat_{repeat_id}.npz', 'wb') as f:
          np.savez(f, results = results_saved)

        self.results_uncompiled.append(results_saved)

        print(f'Finished training repeat {repeat_id} of experiment {self.experiment_id} at {datetime.now()}')

        self.compute_unconditional_irfs(Y_train, Y_test, results, device = device, repeat_id = repeat_id)
        self.compute_multi_forecasts(X_train, X_test, Y_train, Y_test, results, nn_hyps, device, repeat_id)

      # After completing all repeats
      self.is_trained = True

      self._compile_results()

    self._compile_unconditional_irf_results()
    self._compile_multi_forecasting_results()
    self.compute_conditional_irfs()

  # Used when compiling parallel-computed results
  def compile_all(self):
    self.load_results()
    self._compile_results()
    self._compile_multi_forecasting_results()
    self._compile_unconditional_irf_results()

  # Compile results if there are multiple repeats (in results)
  def _compile_results(self):
    
    if self.job_id is not None:
      print('Multiple Jobs, compiling turned off')
      return

    num_repeats = self.run_params['num_repeats']

    repeat_id = 0
    # While there are more repeats to process, stack the fields
    while repeat_id < num_repeats:

      results_repeat = self.results_uncompiled[repeat_id]

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
    self.results = results_compiled
    self.is_compiled = True

    with open(f'{self.folder_path}/params_{self.experiment_id}_compiled.npz', 'wb') as f:
      np.savez(f, results = self.results)

  def _compile_multi_forecasting_results(self):

    if self.execution_params['multi_forecasting'] == False:
      print('Multi Forecasting turned off')
      return 
    if self.job_id is not None:
      print('Multiple Jobs, compiling turned off')
      return
    num_repeats = self.run_params['num_repeats']
    repeat_id = 0
    while repeat_id < num_repeats:

      fcast_repeat = np.load(f'{self.folder_path}/multi_fcast_params_{self.experiment_id}_repeat_{repeat_id}.npz', allow_pickle=True)['fcast']
      if repeat_id == 0:
        fcast_all = fcast_repeat # horizons x variables x bootstraps x time x reestimation_window
      if repeat_id == 0:
        fcast_all = np.concatenate([fcast_all, fcast_repeat], axis = 2)
      repeat_id += 1
    
    with open(f'{self.folder_path}/multi_fcast_{self.experiment_id}_compiled.npz', 'wb') as f:
      np.savez(f, fcast = fcast_all)

  def _compile_unconditional_irf_results(self):

    if self.job_id is not None:
      print('Multiple Jobs, compiling turned off')
      return
    if self.execution_params['unconditional_irfs'] == False:
      print('Unconditional IRFs turned off')
      return 
      
    num_repeats = self.run_params['num_repeats']
    repeat_id = 0
    while repeat_id < num_repeats:

      out_repeat = np.load(f'{self.folder_path}/fcast_params_{self.experiment_id}_repeat_{repeat_id}.npz', allow_pickle=True)
      fcast = out_repeat['fcast']
      fcast_cov_mat = out_repeat['fcast_cov_mat']
      if repeat_id == 0:
        fcast_all = np.zeros((num_repeats, fcast.shape[0], fcast.shape[1], fcast.shape[2], fcast.shape[3]))
        #fcast_cov_mat_all = np.zeros((num_repeats, fcast_cov_mat.shape[0], fcast_cov_mat.shape[1], fcast_cov_mat.shape[2], fcast_cov_mat.shape[3], fcast_cov_mat.shape[4]))
        fcast_cov_mat_all = None
      fcast_all[repeat_id, :,:,:,:] = fcast
      #fcast_cov_mat_all[repeat_id, :,:,:,:,:] = fcast_cov_mat
      repeat_id += 1
    
    with open(f'{self.folder_path}/fcast_{self.experiment_id}_compiled.npz', 'wb') as f:
      np.savez(f, fcast = fcast_all)

    unconditional_irf_params = {
        'n_lag_linear': self.nn_hyps['n_lag_linear'],
        'n_lag_d': self.nn_hyps['n_lag_d'],
        'n_var': len(self.nn_hyps['variables']),
        'num_simulations': self.extensions_params['unconditional_irfs']['num_simulations'],
        'endh': self.extensions_params['unconditional_irfs']['endh'],
        'start_shock_time': self.extensions_params['unconditional_irfs']['start_shock_time'],
        'forecast_method': self.extensions_params['unconditional_irfs']['forecast_method'], # old or new
        'max_h': self.extensions_params['unconditional_irfs']['max_h'], 
        'var_names': self.nn_hyps['variables'],
        'plot_all_bootstraps': self.extensions_params['unconditional_irfs']['plot_all_bootstraps']
      }

    IRFUnconditionalEvaluationObj = IRFUnconditionalEvaluation(fcast_all, fcast_cov_mat_all, unconditional_irf_params)
    IRFUnconditionalEvaluationObj.evaluate_unconditional_irfs()
    IRFUnconditionalEvaluationObj.plot_irfs(self.run_params['image_folder_path'], self.experiment_id)
    IRFUnconditionalEvaluationObj.plot_cumulative_irfs(self.run_params['image_folder_path'], self.experiment_id)

    self.evaluations['unconditional_irf'] = IRFUnconditionalEvaluationObj
    
  def load_results(self):

    # Check if the results exist
    if os.path.exists(f'{self.folder_path}/params_{self.experiment_id}_compiled.npz'):
      self.is_trained = True
      load_file = f'{self.folder_path}/params_{self.experiment_id}_compiled.npz'
      results_loaded = np.load(load_file, allow_pickle = True)['results'].item()
      self.results = results_loaded
      print(f'Loaded compiled results')
      return

    elif os.path.exists(f'{self.folder_path}/params_{self.experiment_id}_repeat_0.npz'):
      repeat_id = 0
      while os.path.exists(f'{self.folder_path}/params_{self.experiment_id}_repeat_{repeat_id}.npz'):
        self.is_trained = True
        load_file = f'{self.folder_path}/params_{self.experiment_id}_repeat_{repeat_id}.npz'
        results_loaded = np.load(load_file, allow_pickle = True)['results'].item()
        self.results_uncompiled.append(results_loaded)

        print(f'Loaded results for repeat {repeat_id}')
        repeat_id += 1

      return
    else:
      print('Not trained yet')

  def __str__(self):
    return f'''
    Run Name: {self.run_name}
    Experiment ID: {self.experiment_id}
    nn_hyps: {self.nn_hyps}
    is_trained: {self.is_trained}
    '''


  
