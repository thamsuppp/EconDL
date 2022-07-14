# Experiment class
from datetime import datetime
import numpy as np
import os

import DataHelpers.DataProcesser as DataProcesser
import TrainVARNN 

keys_to_keep = {'betas': 2,
                'betas_in': 2,
                'sigmas': 3,
                'sigmas_in': 3, 
                'precision': 3,
                'precision_in': 3,
                'cholesky': 4, 
                'cholesky_in': 4, 
                'train_preds': 1, 
                'test_preds': 1}

class Experiment:
  def __init__(self, run_name, experiment_id, nn_hyps, run_params):
    
    self.run_name = run_name
    self.experiment_id = experiment_id
    self.nn_hyps = nn_hyps
    self.run_params = run_params # run_inner_bootstraps, num_repeats, default_nn_hyps

    self.nn_hyps['num_bootstrap'] = self.run_params['num_inner_bootstraps']

    self.results_uncompiled = []
    self.results = []
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

  def get_conditional_irfs(self):
    pass

  # Compile results if there are multiple repeats (in results)
  def _compile_results(self):
    folder_path = self.run_params['folder_path']
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

    with open(f'{folder_path}/params_{self.experiment_id}_compiled.npz', 'wb') as f:
      np.savez(f, results = self.results)

  # @DEV: Don't pass in the dataset in the _init_() because if not then there will be multiply copies of
  # the dataset sitting in each run.
  def train(self, dataset, device):

    if self.is_trained == True:
      print('Trained already')
      return

    X_train, X_test, Y_train, Y_test, nn_hyps = DataProcesser.process_data_wrapper(dataset, self.nn_hyps)
    # For each repeat
    for repeat_id in range(self.run_params['num_repeats']):

      print(nn_hyps)
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

    self.is_trained = True
    self._compile_results()


  def load_results(self):

    folder_path = self.run_params['folder_path']
    # Check if the results exist
    if os.path.exists(f'{folder_path}/params_{self.experiment_id}_compiled.npz'):
      self.is_trained = True
      load_file = f'{folder_path}/params_{self.experiment_id}_compiled.npz'
      results_loaded = np.load(load_file, allow_pickle = True)['results'].item()
      self.results = results_loaded
      print(f'Loaded compiled results')
      return

    elif os.path.exists(f'{folder_path}/params_{self.experiment_id}_repeat_0.npz'):
      repeat_id = 0
      while os.path.exists(f'{folder_path}/params_{self.experiment_id}_repeat_{repeat_id}.npz'):
        self.is_trained = True
        load_file = f'{folder_path}/params_{self.experiment_id}_repeat_{repeat_id}.npz'
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


  
