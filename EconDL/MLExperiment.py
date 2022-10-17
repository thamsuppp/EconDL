from EconDL.Experiment import Experiment
from EconDL.DataHelpers import DataProcesser
from EconDL.ml_benchmark_utils import train_ml_model 
from EconDL.IRF.IRFUnconditional import IRFUnconditional
from EconDL.Forecast.ForecastMulti import ForecastMulti

import numpy as np

class MLExperiment(Experiment):

  def __init__(self, run_name, experiment_id, nn_hyps, run_params, execution_params, extensions_params, job_id = None, reestim_id = None):
    Experiment.__init__(self, run_name, nn_hyps['model'], nn_hyps, run_params, execution_params, extensions_params, job_id, reestim_id)
    
    self.extensions_params = extensions_params
    self.model = nn_hyps['model']
    self.experiment_id = nn_hyps['model']

  def compile_all(self, repeats_to_include = None):
    self._compile_multi_forecasting_results(repeats_to_include = repeats_to_include)
    self._compile_multi_forecasting_reestims()
    self._compile_unconditional_irf_results(repeats_to_include = repeats_to_include)

  def compute_unconditional_irfs(self, Y_train, Y_test, X_train, results, repeat_id):
    unconditional_irf_params = {
        'n_lag_linear': self.nn_hyps['n_lag_linear'],
        'n_lag_d': self.nn_hyps['n_lag_d'],
        'n_var': self.nn_hyps['n_var'],
        'num_simulations': self.extensions_params['unconditional_irfs']['num_simulations'],
        'endh': self.extensions_params['unconditional_irfs']['endh'],
        'start_shock_time': self.extensions_params['unconditional_irfs']['start_shock_time'],
        'forecast_method': self.extensions_params['unconditional_irfs']['forecast_method'], # old or new
        'max_h': self.extensions_params['unconditional_irfs']['max_h'], 
        'var_names': self.nn_hyps['var_names'],
        'model': self.model,
        'end_precision_lambda': 0
      }

    IRFUnconditionalObj = IRFUnconditional(unconditional_irf_params, None)
    fcast, _, sim_shocks = IRFUnconditionalObj.get_irfs_wrapper(Y_train, Y_test, X_train, results)

    with open(f'{self.folder_path}/fcast_params_{self.model}_repeat_{repeat_id}_reestim_{self.reestim_id}.npz', 'wb') as f:
      np.savez(f, fcast = fcast)

  def compute_multi_forecasts(self, X_train, X_test, Y_train, Y_test, results, nn_hyps, repeat_id):
    multi_forecasting_params = {
      'test_size': self.nn_hyps['test_size'], 
      'forecast_horizons': self.extensions_params['multi_forecasting']['forecast_horizons'],
      'reestimation_window': self.extensions_params['multi_forecasting']['reestimation_window'],
      'num_inner_bootstraps': self.nn_hyps['num_inner_bootstraps'],
      'num_sim_bootstraps': self.extensions_params['multi_forecasting']['num_sim_bootstraps'],
      'num_repeats': 1, 

      'n_lag_linear': self.nn_hyps['n_lag_linear'],
      'n_lag_d': self.nn_hyps['n_lag_d'],
      'n_var': self.nn_hyps['n_var'],
      'forecast_method': self.extensions_params['multi_forecasting']['forecast_method'], # old or new
      'var_names': self.nn_hyps['var_names'],
      'model': self.model,
      'end_precision_lambda': 0
    }

    ForecastMultiObj = ForecastMulti(self.run_name, Y_train, Y_test, multi_forecasting_params, device = None)
    FCAST = ForecastMultiObj.conduct_multi_forecasting_wrapper(X_train, X_test, results, nn_hyps)

    print('Done with Multiforecasting')
    with open(f'{self.folder_path}/multi_fcast_params_{self.model}_repeat_{repeat_id}_reestim_{self.reestim_id}.npz', 'wb') as f:
      np.savez(f, fcast = FCAST)

  def train(self, dataset):
    
    if self.is_trained == True:
      print('Trained already')
    else:
      print(self.nn_hyps)
      # nn_hyps needed: test_size, variables, num_bootstrap, time_dummy_setting, marx, dummy_interval, s_pos, s_pos_setting, model
      X_train, X_test, Y_train, Y_test, nn_hyps = DataProcesser.process_data_wrapper(dataset, self.nn_hyps)
      
      repeat_ids = []
      if self.job_id is not None:
        repeat_ids = [self.job_id]
      else:
        repeat_ids = range(self.run_params['num_repeats'])

      # For each repeat
      for repeat_id in repeat_ids:

        results = train_ml_model(X_train, X_test, Y_train, Y_test, nn_hyps)

        self.compute_unconditional_irfs(Y_train, Y_test, X_train, results, repeat_id = repeat_id)
        self.compute_multi_forecasts(X_train, X_test, Y_train, Y_test, results, nn_hyps, repeat_id)

      # After completing all repeats
      self.is_trained = True

      self._compile_results()
    self._compile_unconditional_irf_results()
    self._compile_multi_forecasting_results()
