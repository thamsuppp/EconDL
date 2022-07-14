import numpy as np
import pandas as pd
import torch
import DataHelpers.DataLoader as DataLoader
import DataHelpers.DataProcesser as DataProcesser # possible to combine w dataloader
import IRF.IRFConditional as IRFConditional # @TODO: fix this import (want to put it in a folder) - also create IRF superclass, and IRFConditional and IRFUnconditional subclasses
from IRF.IRFUnconditional import IRFUnconditional
import TrainVARNN 
from Benchmarks import Benchmarks # @TODO: create Benchmark superclass, and VARNNBenchmarks and ForecastBenchmarks subclasses
import ForecastBenchmarks 
import ForecastMulti
import Evaluation

from Run import Run
from Experiment import Experiment

import json
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment name is the command-line argument
run_name = sys.argv[1]

# Instantiate the Run:
# - Creates a new folder
# - Reads the json
# - Instantiates the experiments with the corresponding nn_hyps
# - Loads the data into the RunObj
RunObj = Run(run_name, device)
RunObj.print_params()
# Train all experiments within the run and store the experiments within the object
RunObj.train_experiments()

# Everything above this works!






#     if run_params['execution_params']['unconditional_irfs'] == True:

#       unconditional_irf_params = {
#         'n_lag_linear': nn_hyps['n_lag_linear'],
#         'n_lag_d': nn_hyps['n_lag_d'],
#         'n_var': len(nn_hyps['variables']),
#         'num_simulations': 600,
#         'endh': 40,
#         'start_shock_time': 40,
#         'forecast_method': 'new', # old or new
#         'max_h': 20, 
#         'var_names': nn_hyps['variables'],
#         'plot_all_bootstraps': False
#       }

#       IRFUnconditionalObj = IRFUnconditional(run_name, unconditional_irf_params, device)
#       fcast, fcast_cov_mat, sim_shocks = IRFUnconditionalObj.get_irfs_wrapper(Y_train, Y_test, results)

#       with open(f'{folder_path}/fcast_params_{experiment_id}_repeat{repeat_id}.npz', 'wb') as f:
#         np.savez(f, fcast = fcast, fcast_cov_mat = fcast_cov_mat)

#     if run_params['execution_params']['multi_forecasting'] == True:

#       multi_forecasting_params = {
#         'test_size': 60, 
#         'forecast_horizons': 6,
#         'reestimation_window': 60,
#         'num_inner_bootstraps': num_inner_bootstraps,
#         'num_repeats': 1, 

#         'n_lag_linear': nn_hyps['n_lag_linear'],
#         'n_lag_d': nn_hyps['n_lag_d'],
#         'n_var': len(nn_hyps['variables']),
#         'forecast_method': 'new', # old or new
#         'var_names': nn_hyps['variables'],
#       }


# # Compute benchmarks
# benchmark_params = {
#   'n_lag_linear': 1, 
#   'n_lag_d': 2,
#   'benchmarks': ['VAR_whole', 'AR_whole', 'VAR_roll', 'AR_roll', 'VAR_expand', 'AR_expand', 'RF_whole'],
#   'var_names': ['DGS3', 'inf', 'unrate'],
#   'test_size': 100,
#   'window_length': 40,
#   'reestimation_window': 1
# }
# if run_params['execution_params']['benchmarks'] == True:
#   BenchmarkObj = Benchmarks(dataset, benchmark_params, run_name)
#   BenchmarkObj.compute_benchmarks()


# # Compute conditional IRFs (straight from VARNN estimation results) and plot
# # @DEV: do conditional IRFs come from 
# if run_params['execution_params']['conditional_irfs'] == True:
#   irf_cond_results = IRFConditional.compute_IRF()
