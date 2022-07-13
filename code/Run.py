# Run class
import json
from Experiment import Experiment

class Run:

  def __init__(self, run_name):

    self.run_name = run_name

    self.dataset = None
    self.run_params = None
    self.execution_params = None
    self.experiment_params = None

    self.experiments = []

    self._load_params()
    self._init_experiments()

  def print_params(self):
    print('Run Params: ', self.run_params)
    print('Execution Params: ', self.execution_params)
    print('Experiment Params: ', self.experiment_params)

    print('Experiment Objs: ')
    for exp in self.experiments:
      print(exp)
  
  def _load_params(self):
    with open(f'../exp_config/{self.run_name}.json', 'r') as f:
      all_params = json.load(f)
    
    self.run_params = all_params['run_params']
    self.execution_params = all_params['execution_params']
    self.experiment_params = all_params['nn_hyps']

  def _init_experiments(self): # Only if experiments are not already initialized
    if self.experiments == []:
      for experiment_id, nn_hyps in enumerate(self.experiment_params):
        ExperimentObj = Experiment(self.run_name, experiment_id, nn_hyps)
        self.experiments.append(ExperimentObj)

