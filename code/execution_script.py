import numpy as np
import pandas as pd
import DataLoader
import DataProcesser # possible to combine w dataloader
import IRF.IRFConditional as IRFConditional # @TODO: fix this import (want to put it in a folder) - also create IRF superclass, and IRFConditional and IRFUnconditional subclasses
import IRF.IRFUnconditional as IRFUnconditional
import TrainVARNN 
import Benchmarks # @TODO: create Benchmark superclass, and VARNNBenchmarks and ForecastBenchmarks subclasses
import ForecastBenchmarks 
import ForecastMulti
import Evaluation

import json
import os
import sys

# Experiment name is the command-line argument
experiment_name = sys.argv[1]

# Read experiment configuration - dataset, parameters: nn_hyps, experiment_settings (num_repeats), evaluation_settings (how to draw graphs etc.)
with open(f'../exp_config/{experiment_name}.json', 'r') as f:
    experiment_params = json.load(f)

# Create folder to store results - that is where the results go into
folder_path = f'../results/{experiment_name}'
if os.path.isdir(folder_path) == False:
  os.mkdir(folder_path)
else:
  print('Folder already exists')

# Add experiment to a running list of experiments I've run

# Load dataset
dataset, experiment_params = DataLoader.load_data(experiment_params)

print(dataset.head())
print(experiment_params['n_var'])
print(experiment_params['var_names'])

# Process dataset
processed_dataset = DataProcesser.process_data(dataset)

# Train the VARNN
results = TrainVARNN.train(processed_dataset, experiment_params)

# Save the training results

# Compute conditional IRFs (straight from VARNN estimation results) and plot
# @DEV: do conditional IRFs come from 
irf_cond_results = IRFConditional.compute_IRF()

# Compute unconditional IRFs (only if there is no time hemisphere!) and plot
irf_uncond_results = IRFUnconditional.compute_IRF()

# Compute benchmarks
benchmark_results = Benchmarks.compute_benchmarks()

# Do multi-horizon forecasting
fcast_results = ForecastMulti.get_forecasts()

# Compute benchmarks for multi-horizon forecasting
fcast_benchmark_results = ForecastBenchmarks.compute_benchmarks()

# Evaluation - plot TVPs, betas and sigmas
Evaluation.evaluate_TVPs(results, benchmark_results)

# Evaluation - one-step forecasting - compute MSEs (plot graphs of cumulative MSEs)
Evaluation.evaluate_one_step_forecasts(results, benchmark_results)

# Evaluation - multi-step forecasting
Evaluation.evaluate_multi_step_forecasts(results, benchmark_results)