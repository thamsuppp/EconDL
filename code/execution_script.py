import numpy as np
import pandas as pd
import DataLoader
import DataProcesser # possible to combine w dataloader
import IRFConditional # @TODO: fix this import (want to put it in a folder) - also create IRF superclass, and IRFConditional and IRFUnconditional subclasses
import IRFUnconditional
import TrainVARNN 
import Benchmarks # @TODO: create Benchmark superclass, and VARNNBenchmarks and ForecastBenchmarks subclasses
import ForecastBenchmarks 
import ForecastMulti
import Evaluation

import json

# @DEV: what is the accepted way to read config for python (or anything) from industry? store in another constants.py (same language) file, or read from JSON, or what?

# Read experiment configuration - dataset, parameters: nn_hyps, experiment_settings (num_repeats), evaluation_settings (how to draw graphs etc.)

# @DEV: somehow does not work (ConnectionRefusedError)
experiment_params = json.load('exp_config/exp1_config.json')

print(experiment_params)

# Create folder to store results - that is where the results go into
# Add experiment to a running list of experiments I've run

# Load dataset
dataset = DataLoader.load_data()

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