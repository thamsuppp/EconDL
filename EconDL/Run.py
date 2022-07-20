# Run class
import json
import os
from EconDL.Experiment import Experiment
from EconDL.Benchmarks import Benchmarks
import EconDL.DataHelpers.DataLoader as DataLoader
from EconDL.Forecast.ForecastBenchmarks import ForecastBenchmarks
from EconDL.MLExperiment import MLExperiment

class Run:
  
  def __init__(self, run_name, device, experiment_id = None, job_id = None):

    self.run_name = run_name
    self.device = device
    self.job_id = job_id
    self.experiment_id = experiment_id

    self.dataset_name = None
    self.dataset = None
    self.n_var = None
    self.var_names = None
    self.run_params = None
    self.execution_params = None
    self.extensions_params = None
    self.experiment_params = None
    self.evaluation_params = None

    self.experiments = []
    self.num_experiments = 0

    self.ml_experiments = []

    self._load_params()

    # Create folder to store results - that is where the results go into
    self.folder_path = f'results/{run_name}'
    if os.path.isdir(self.folder_path) == False:
      os.mkdir(self.folder_path)
      print(f'Folder made at {self.folder_path}')
    else:
      print(f'Folder at {self.folder_path} already exists')

    # Create image folder if not exist yet
    self.image_folder_path = f'{self.folder_path}/images'
    if os.path.isdir(self.image_folder_path) == False:
      os.mkdir(self.image_folder_path)

    self.run_params.update(
      {
        'folder_path': self.folder_path,
        'image_folder_path': self.image_folder_path
        }
      )
    
    self._init_experiments()
    self._init_ml_experiments()
    self._load_data()

  def _load_params(self):
    with open(f'exp_config/{self.run_name}.json', 'r') as f:
      all_params = json.load(f)
    
    self.run_params = all_params['run_params']
    self.execution_params = all_params['execution_params']
    self.experiment_params = all_params['nn_hyps']
    self.extensions_params = all_params['extensions_params']
    self.evaluation_params = all_params['evaluation_params']
    self.dataset_name = all_params['run_params']['dataset']

    self.n_var = all_params['run_params']['n_var']
    self.var_names = all_params['run_params']['var_names']

    # If specific experiment ID is specified, filter for that experiment params
    if self.experiment_id is not None:
      self.experiment_params = [self.experiment_params[self.experiment_id]]

      print(f'Run _load_params(): Load params for only Expeirment {self.experiment_id}')

  def _init_ml_experiments(self):

    # Load the default nn_hyps
    default_nn_hyps_path = self.run_params['default_nn_hyps']
    with open(f'exp_config/{default_nn_hyps_path}.json', 'r') as f:
      default_nn_hyps = json.load(f)

    # Combine default_nn_hyps with the run_params
    default_nn_hyps.update(self.run_params)

    for ml_model in self.extensions_params['ml_experiments']:
      all_nn_hyps = default_nn_hyps.copy()
      all_nn_hyps.update({'model': ml_model})

      MLExperimentObj = MLExperiment(self.run_name, None, all_nn_hyps, self.run_params, self.execution_params, self.extensions_params, self.job_id)
      self.ml_experiments.append(MLExperimentObj)

  def _init_experiments(self): # Only if experiments are not already initialized

    # Load the default nn_hyps
    default_nn_hyps_path = self.run_params['default_nn_hyps']
    with open(f'exp_config/{default_nn_hyps_path}.json', 'r') as f:
      default_nn_hyps = json.load(f)

    # Combine default_nn_hyps with the run_params
    default_nn_hyps.update(self.run_params)

    # If this run is initialized with a specific Experiment ID
    if self.experiment_id is not None:
      all_nn_hyps = default_nn_hyps.copy()
      all_nn_hyps.update(self.experiment_params[0])
      all_nn_hyps['model'] = 'VARNN'
      ExperimentObj = Experiment(self.run_name, self.experiment_id, all_nn_hyps, self.run_params, self.execution_params, self.extensions_params, self.job_id)
      self.experiments.append(ExperimentObj)
      self.num_experiments = 1

    if self.experiments == []:
      for experiment_id, changed_nn_hyps in enumerate(self.experiment_params):
        # Combine the default nn_hyps with changed_nn_hyps
        all_nn_hyps = default_nn_hyps.copy()
        all_nn_hyps.update(changed_nn_hyps)
        all_nn_hyps['model'] = 'VARNN'
        ExperimentObj = Experiment(self.run_name, experiment_id, all_nn_hyps, self.run_params, self.execution_params, self.extensions_params, self.job_id)
        self.experiments.append(ExperimentObj)
      
      self.num_experiments = len(self.experiments)
  
  def _load_data(self):
    self.dataset, _, _ = DataLoader.load_data(self.dataset_name)

  def train_experiments(self, experiment_ids = None):  
    # If experiment_ids = None, then train all
    experiment_ids = experiment_ids if experiment_ids else list(range(self.num_experiments))
    
    for experiment_id in experiment_ids:
      ExperimentObj = self.experiments[experiment_id]
      ExperimentObj.train(self.dataset, self.device)

  def train_ml_experiments(self):
    for MLExperiment in self.ml_experiments:
      MLExperiment.train(self.dataset)

  def compile_experiments(self, experiment_ids = None, repeats_to_include = None):  
    # If experiment_ids = None, then train all
    print(f'Run compile_experiments(): repeats_to_include {repeats_to_include}')
    experiment_ids = experiment_ids if experiment_ids else list(range(self.num_experiments))
    for experiment_id in experiment_ids:
      ExperimentObj = self.experiments[experiment_id]
      ExperimentObj.load_results(repeats_to_include = repeats_to_include)
      ExperimentObj.compile_all(repeats_to_include = repeats_to_include)

  def compile_ml_experiments(self, repeats_to_include = None):
    for MLExperimentObj in self.ml_experiments:
      MLExperimentObj.compile_all(repeats_to_include = repeats_to_include)

  def train_benchmarks(self):

    # Check if the results exist == benchmark_folder exists
    if os.path.isdir(f'{self.folder_path}/benchmarks'):
      print('Benchmarks already trained')
    else:

      print('Training Benchmarks')
      benchmark_params = {
        'n_lag_linear': self.run_params['n_lag_linear'], 
        'n_lag_d': self.run_params['n_lag_d'],
        'benchmarks': ['VAR_whole', 'AR_whole', 'VAR_roll', 'AR_roll', 'VAR_expand', 'AR_expand', 'RF_whole'],
        'var_names': self.run_params['var_names'],
        'test_size': self.run_params['test_size'],
        'window_length': self.extensions_params['benchmarks']['window_length'],
        'reestimation_window': self.extensions_params['benchmarks']['reestimation_window']
      }

      BenchmarkObj = Benchmarks(self.dataset, benchmark_params, self.run_name)
      BenchmarkObj.compute_benchmarks()

      print('Benchmarks trained')

  def train_multi_forecast_benchmarks(self):
    # # Check if the results exist == benchmark_folder exists
    # if os.path.isdir(f'{self.folder_path}/benchmarks'):
    #   print('Benchmarks already trained')
    if self.execution_params['multi_forecasting'] == False:
      print('Multiforecasting turned off')
      return

    print('Training Multi-forecasting Benchmarks')
    multi_forecasting_params = {
      'test_size': self.run_params['test_size'], 
      'forecast_horizons': self.extensions_params['multi_forecasting']['forecast_horizons'],
      'reestimation_window': self.extensions_params['multi_forecasting']['reestimation_window'],
      'num_inner_bootstraps': self.run_params['num_inner_bootstraps'],
      'num_sim_bootstraps': self.extensions_params['multi_forecasting']['num_sim_bootstraps'],
      'benchmarks': self.extensions_params['multi_forecasting']['benchmarks'],
      'num_repeats': 1, 

      'n_lag_linear': self.run_params['n_lag_linear'],
      'n_lag_d': self.run_params['n_lag_d'],
      'n_var': self.n_var,
      'forecast_method': self.extensions_params['multi_forecasting']['forecast_method'], # old or new
      'var_names': self.var_names
    }

    ForecastBenchmarkObj = ForecastBenchmarks(self.dataset, multi_forecasting_params, self.run_name)
    ForecastBenchmarkObj.conduct_multi_forecasting_benchmarks()

    print('Multi-forecasting Benchmarks trained')

  def compute_conditional_irfs(self, experiment_ids = None):
    # If experiment_ids = None, then train all
    experiment_ids = experiment_ids if experiment_ids else list(range(self.num_experiments))
    for experiment_id in experiment_ids:
      ExperimentObj = self.experiments[experiment_id]
      ExperimentObj.compute_conditional_irfs()

  # Wrapper function that trains experiments, benchmarks and multi-forecast benchmarks
  def train_all(self):
    self.train_experiments()
    self.train_benchmarks()
    self.train_multi_forecast_benchmarks()

  def print_params(self):
    print('Run Params: ', self.run_params)
    print('Execution Params: ', self.execution_params)
    print('Experiment Params: ', self.experiment_params)

    print('Experiment Objs: ')
    for exp in self.experiments:
      print(exp)

  def print_data(self):
    if self.dataset is not None:
      print(f'Variables: {self.var_names}')
      print(f'Dataset Shape: {self.dataset.shape}')
      print(self.dataset.head())
    else:
      print('No dataset')