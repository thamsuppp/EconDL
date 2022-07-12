import numpy as np
import pandas as pd
import torch
import DataHelpers.DataLoader as DataLoader
import DataHelpers.DataProcesser as DataProcesser # possible to combine w dataloader
import IRF.IRFConditional as IRFConditional # @TODO: fix this import (want to put it in a folder) - also create IRF superclass, and IRFConditional and IRFUnconditional subclasses
import IRF.IRFUnconditional as IRFUnconditional
import TrainVARNN 
from Benchmarks import Benchmarks # @TODO: create Benchmark superclass, and VARNNBenchmarks and ForecastBenchmarks subclasses
import ForecastBenchmarks 
import ForecastMulti
import Evaluation

import json
import os
import sys

from nn_hyps import nn_hyps_default


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment name is the command-line argument
#run_name = sys.argv[1]
run_name = '10jul_test'

print(os.getcwd())

# Read experiment configuration - dataset, parameters: nn_hyps, experiment_settings (num_repeats), evaluation_settings (how to draw graphs etc.)
with open(f'../exp_config/{run_name}.json', 'r') as f:
    run_params = json.load(f)

# Create folder to store results - that is where the results go into
folder_path = f'../results/{run_name}'
if os.path.isdir(folder_path) == False:
  os.mkdir(folder_path)
else:
  print('Folder already exists')

# Add experiment to a running list of experiments I've run

# Load dataset
dataset, run_params = DataLoader.load_data(run_params)

# Get the number of experiments to run
num_experiments = len(run_params['nn_hyps'])
num_repeats = run_params['run_params']['num_repeats']
num_inner_bootstraps = run_params['run_params']['num_inner_bootstraps']

if run_params['execution_params']['varnn_estimation'] == True:

  for repeat_id in range(num_repeats):
    for experiment_id in range(num_experiments):

      experiment_params = run_params['nn_hyps'][experiment_id]

      print(f'Experiment {experiment_id}, Params: {experiment_params}')
      nn_hyps = nn_hyps_default.copy()
      nn_hyps.update(experiment_params)
      nn_hyps['num_bootstrap'] = num_inner_bootstraps

      # Process dataset - DONE
      X_train, X_test, Y_train, Y_test, nn_hyps = DataProcesser.process_data_wrapper(dataset, nn_hyps)

      # Train the VARNN
      results = TrainVARNN.conduct_bootstrap(X_train, X_test, Y_train, Y_test, nn_hyps, device)

      # Save the training results
      BETAS = results['betas_draws']
      BETAS_IN = results['betas_in_draws']
      SIGMAS = results['sigmas_draws']
      SIGMAS_IN = results['sigmas_in_draws']
      PRECISION = results['precision_draws']
      PRECISION_IN = results['precision_in_draws']
      CHOLESKY = results['cholesky_draws']
      CHOLESKY_IN = results['cholesky_in_draws']
      PREDS = results['pred_in_ensemble'] 
      PREDS_TEST = results['pred_ensemble']

      with open(f'{folder_path}/params_{experiment_id}_repeat_{repeat_id}.npz', 'wb') as f:
          np.savez(f, betas = BETAS, betas_in = BETAS_IN, 
              sigmas = SIGMAS, sigmas_in = SIGMAS_IN,
              precision = PRECISION, precision_in = PRECISION_IN,
              cholesky = CHOLESKY, cholesky_in = CHOLESKY_IN,
              train_preds = PREDS, test_preds = PREDS_TEST, 
              y = Y_train, y_test = Y_test, 
              params = nn_hyps)

# Compute benchmarks
benchmark_params = {
  'n_lag_linear': 2,
  'n_lag_d': 8,
  'benchmarks': ['VAR_whole', 'AR_whole', 'VAR_roll', 'AR_roll', 'VAR_expand', 'AR_expand', 'RF_whole'],
  'var_names': ['DGS3', 'inf', 'unrate'],
  'test_size': 90,
  'window_length': 80,
  'reestimation_window': 1
}
if run_params['execution_params']['benchmarks'] == True:
  BenchmarkObj = Benchmarks(dataset, benchmark_params, run_name)
  BenchmarkObj.compute_benchmarks()


# # Compute conditional IRFs (straight from VARNN estimation results) and plot
# # @DEV: do conditional IRFs come from 
# irf_cond_results = IRFConditional.compute_IRF()

# # Compute unconditional IRFs (only if there is no time hemisphere!) and plot
# irf_uncond_results = IRFUnconditional.compute_IRF()

# # Compute benchmarks
# benchmark_results = Benchmarks.compute_benchmarks()

# # Do multi-horizon forecasting
# fcast_results = ForecastMulti.get_forecasts()

# # Compute benchmarks for multi-horizon forecasting
# fcast_benchmark_results = ForecastBenchmarks.compute_benchmarks()

# # Evaluation - plot TVPs, betas and sigmas
# Evaluation.evaluate_TVPs(results, benchmark_results)

# # Evaluation - one-step forecasting - compute MSEs (plot graphs of cumulative MSEs)
# Evaluation.evaluate_one_step_forecasts(results, benchmark_results)

# # Evaluation - multi-step forecasting
# Evaluation.evaluate_multi_step_forecasts(results, benchmark_results)