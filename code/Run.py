# Run class
import json
import os
from Experiment import Experiment
from Benchmarks import Benchmarks
import DataHelpers.DataLoader as DataLoader


class Run:
  
  def __init__(self, run_name, device):

    self.run_name = run_name
    self.device = device

    self.dataset_name = None
    self.dataset = None
    self.n_var = None
    self.var_names = None
    self.run_params = None
    self.execution_params = None
    self.extensions_params = None
    self.experiment_params = None

    self.experiments = []
    self.num_experiments = 0

    self._load_params()

    # Create folder to store results - that is where the results go into
    self.folder_path = f'../results/{run_name}'
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
    self._load_data()


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

  def _load_params(self):
    with open(f'../exp_config/{self.run_name}.json', 'r') as f:
      all_params = json.load(f)
    
    self.run_params = all_params['run_params']
    self.execution_params = all_params['execution_params']
    self.experiment_params = all_params['nn_hyps']
    self.extensions_params = all_params['extensions_params']
    self.dataset_name = all_params['dataset']

  def _init_experiments(self): # Only if experiments are not already initialized

    # Load the default nn_hyps
    default_nn_hyps_path = self.run_params['default_nn_hyps']
    with open(f'../exp_config/{default_nn_hyps_path}.json', 'r') as f:
      default_nn_hyps = json.load(f)

    # Combine default_nn_hyps with the run_params
    default_nn_hyps.update(self.run_params)

    if self.experiments == []:
      for experiment_id, changed_nn_hyps in enumerate(self.experiment_params):
        # Combine the default nn_hyps with changed_nn_hyps
        all_nn_hyps = default_nn_hyps.copy()
        all_nn_hyps.update(changed_nn_hyps)
        ExperimentObj = Experiment(self.run_name, experiment_id, all_nn_hyps, self.run_params, self.execution_params, self.extensions_params)
        self.experiments.append(ExperimentObj)
      
      self.num_experiments = len(self.experiments)
  
  def _load_data(self):
    self.dataset, self.n_var, self.var_names = DataLoader.load_data(self.dataset_name)

  def train_experiments(self, experiment_ids = None):  
    # If experiment_ids = None, then train all
    experiment_ids = experiment_ids if experiment_ids else list(range(self.num_experiments))
    
    for experiment_id in experiment_ids:
      ExperimentObj = self.experiments[experiment_id]
      ExperimentObj.train(self.dataset, self.device)

  def get_conditional_irfs(self, experiment_ids = None):
    # If experiment_ids = None, then train all
    experiment_ids = experiment_ids if experiment_ids else list(range(self.num_experiments))
    
    for experiment_id in experiment_ids:
      ExperimentObj = self.experiments[experiment_id]
      ExperimentObj.get_conditional_irfs()


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