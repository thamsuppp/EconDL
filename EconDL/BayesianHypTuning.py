# Class that does Bayesian hyperparameter tuning

import json
import os 
import numpy as np

from EconDL.Experiment import Experiment
import EconDL.DataHelpers.DataLoader as DataLoader
from EconDL.utils import get_bootstrap_indices

from skopt import gp_minimize
from skopt import callbacks, dump, load
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

class BayesianHypTuning:

  def __init__(self, run_name, device, dimensions, dim_names, default_parameters):

    self.run_name = run_name
    self.device = device

    self.dataset_name = None
    self.dataset = None
    self.exog_dataset = None
    self.n_var = None
    self.var_names = None
    self.default_nn_hyps = None
    self.run_params = None
    self.execution_params = None
    self.extensions_params = None
    self.experiment_params = None
    self.evaluation_params = None

    self.dimensions = dimensions
    self.dim_names = dim_names
    self.default_parameters = default_parameters

    self.n_bayesian_iterations = None

    self.experiments = []
    self.num_experiments = 0

    self.experiment_records = []

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

    self._load_data()

    train_size = self.dataset.shape[0] - self.run_params['test_size'] - self.run_params['n_lag_d']

    self.bootstrap_indices = get_bootstrap_indices(num_bootstrap = self.run_params['num_inner_bootstraps'], 
                        n_obs = train_size, block_size = self.default_nn_hyps['block_size'], 
                        sampling_rate = self.default_nn_hyps['sampling_rate'], 
                        opt_bootstrap = self.default_nn_hyps['opt_bootstrap'])

    
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
    self.n_bayesian_iterations = all_params['run_params'].get('n_bayesian_iterations', 50)

    # Load the default nn_hyps
    default_nn_hyps_path = self.run_params['default_nn_hyps']
    with open(f'exp_config/{default_nn_hyps_path}.json', 'r') as f:
      self.default_nn_hyps = json.load(f)

  def _load_data(self):
    self.dataset, _, _, self.exog_dataset = DataLoader.load_data(self.dataset_name)

  # Init the experiment, setting the hyperparameters
  def init_experiment(self, hyperparameters, experiment_id):

    default_nn_hyps = self.default_nn_hyps.copy()
    # Combine default_nn_hyps with the run_params
    default_nn_hyps.update(self.run_params)
    # Update for the current experiment's hyperparameters
    default_nn_hyps.update(hyperparameters)
    default_nn_hyps['model'] = "VARNN"
    # Update the bootstrap indices
    default_nn_hyps['bootstrap_indices'] = self.bootstrap_indices

    print('Hyperparameters: ', default_nn_hyps)

    ExperimentObj = Experiment(self.run_name, experiment_id, default_nn_hyps, self.run_params, self.execution_params, self.extensions_params, None)
    self.experiments.append(ExperimentObj)
    return ExperimentObj

  # Function that trains experiment
  def train_experiment(self, ExperimentObj):

    # Train the experiment
    ExperimentObj.train(self.dataset, self.exog_dataset, self.device)
    results = ExperimentObj.results_uncompiled[0]

    train_preds = results['train_preds']
    test_preds = results['test_preds']
    Y_train = results['y']
    Y_test = results['y_test']

    # Get the median prediction across all bootstraps
    train_preds_median = np.nanmedian(train_preds, axis = 1)
    train_error = np.abs(Y_train - train_preds_median)
    # Mean Error (across all variables)
    train_mean_error = np.nanmean(train_error, axis = 0)
    # Divide by each variable's SD (to make each variable be worth the same)
    train_mean_error = train_mean_error / np.std(Y_train, axis = 0)

    # Get the test error
    test_preds_median = np.nanmedian(test_preds, axis = 1)
    test_error = np.abs(Y_test - test_preds_median)
    test_mean_error = np.nanmean(test_error, axis = 0)
    test_mean_error = test_mean_error / np.std(Y_test, axis = 0)

    # Save the results (evaluate the experiment?)
    return train_mean_error, test_mean_error

  def fitness(self, **kwargs):
    # Create a dictionary to contain the kwargs
    new_params = {
      'dropout_rate': kwargs['dropout_rate'],
      'nodes': [kwargs['nn_width'] for e in range(kwargs['nn_depth'])],
      'tvpl_archi': [kwargs['tvpl']],
      'constant_tvpl': [kwargs['constant_tvpl']],
      'precision_lambda': kwargs['precision_lambda'],
      'lr': kwargs['lr'],
      'activation': kwargs['activation'],
    }
    print(f'Experiment is {self.num_experiments}, new params are {new_params}')

    ExperimentObj = self.init_experiment(new_params, self.num_experiments)
    train_mean_error, test_mean_error = self.train_experiment(ExperimentObj) # returns the standardized MAE for all variables
    score = np.sum(train_mean_error) # mean across all variables
    print('Mean Error: ', train_mean_error, 'Test Mean Error: ', test_mean_error, 'Score:', score)

    # Save the record
    self.experiment_records.append({
      'experiment_id': self.num_experiments,
      'train_mean_error': train_mean_error,
      'test_mean_error': test_mean_error,
      'params': new_params
    })
    # Increment num_experiments
    self.num_experiments += 1

    # Get the OOB error and test error
    return score

  # Overall wrapper function that does the Bayesian hyperparameter tuning
  def conduct_hyp_tuning(self):

    # TODO: Fitness function - takes in hyperparameters by argument, and returns score (error)
    @use_named_args(dimensions = self.dimensions)
    def fitness_wrapper(*args, **kwargs):
      return self.fitness(*args, **kwargs)

    search_result = gp_minimize(
      func = fitness_wrapper,
      dimensions = self.dimensions,
      acq_func = 'EI', # Expected Improvement
      n_calls = self.n_bayesian_iterations, 
      x0 = self.default_parameters,
      random_state = 42
    )

    dump(search_result, f"{self.folder_path}/opt_results.pkl", store_objective = False)
    with open(f'{self.folder_path}/experiment_records.npz', 'wb') as f:
      np.savez(f, experiment_records = self.experiment_records)


