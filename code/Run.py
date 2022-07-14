# Run class
import json
import os
from Experiment import Experiment
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
    self.experiment_params = None

    self.experiments = []
    self.num_experiments = 0

    self._load_params()

    # Create folder to store results - that is where the results go into
    folder_path = f'../results/{run_name}'
    if os.path.isdir(folder_path) == False:
      os.mkdir(folder_path)
      print(f'Folder made at {folder_path}')
    else:
      print(f'Folder at {folder_path} already exists')

    self.run_params.update({'folder_path': folder_path})

    
    self._init_experiments()
    self._load_data()


  def _load_params(self):
    with open(f'../exp_config/{self.run_name}.json', 'r') as f:
      all_params = json.load(f)
    
    self.run_params = all_params['run_params']
    self.execution_params = all_params['execution_params']
    self.experiment_params = all_params['nn_hyps']
    self.dataset_name = all_params['dataset']

  def _init_experiments(self): # Only if experiments are not already initialized

    # Load the default nn_hyps
    default_nn_hyps_path = self.run_params['default_nn_hyps']
    with open(f'../exp_config/{default_nn_hyps_path}.json', 'r') as f:
      default_nn_hyps = json.load(f)

    if self.experiments == []:
      for experiment_id, changed_nn_hyps in enumerate(self.experiment_params):
        # Combine the default nn_hyps with changed_nn_hyps
        all_nn_hyps = default_nn_hyps.copy()
        default_nn_hyps.update(changed_nn_hyps)
        ExperimentObj = Experiment(self.run_name, experiment_id, all_nn_hyps, self.run_params)
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