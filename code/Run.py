# Run class
import json
import os
from Experiment import Experiment
import DataHelpers.DataLoader as DataLoader


class Run:

  def __init__(self, run_name):

    self.run_name = run_name

    self.dataset_name = None
    self.dataset = None
    self.n_var = None
    self.var_names = None
    self.run_params = None
    self.execution_params = None
    self.experiment_params = None

    self.experiments = []

    self._load_params()
    self._init_experiments()
    self._load_data()

    # Create folder to store results - that is where the results go into
    folder_path = f'../results/{run_name}'
    if os.path.isdir(folder_path) == False:
      os.mkdir(folder_path)
      print(f'Folder made at {folder_path}')
    else:
      print(f'Folder at {folder_path} already exists')

  def _load_params(self):
    with open(f'../exp_config/{self.run_name}.json', 'r') as f:
      all_params = json.load(f)
    
    self.run_params = all_params['run_params']
    self.execution_params = all_params['execution_params']
    self.experiment_params = all_params['nn_hyps']
    self.dataset_name = all_params['dataset']

  def _init_experiments(self): # Only if experiments are not already initialized
    if self.experiments == []:
      for experiment_id, nn_hyps in enumerate(self.experiment_params):
        ExperimentObj = Experiment(self.run_name, experiment_id, nn_hyps)
        self.experiments.append(ExperimentObj)
  
  def _load_data(self):
    self.dataset, self.n_var, self.var_names = DataLoader.load_data(self.dataset_name)


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