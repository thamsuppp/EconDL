import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from IRF.IRFConditional import IRFConditional
import Evaluation


# Experiment name is the command-line argument
experiment_name = sys.argv[1]

folder_path = f'../results/{experiment_name}'


# Create image folder if not exist yet
image_folder_path = f'{folder_path}/images'
if os.path.isdir(image_folder_path) == False:
  os.mkdir(image_folder_path)

# Load the saved VARNN model output
results = np.load(f'{folder_path}/params.npz', allow_pickle = True)
BETAS = results['betas']
BETAS_IN = results['betas_in']
PREDS = results['train_preds']
PREDS_TEST = results['test_preds']
SIGMAS = results['sigmas']
SIGMAS_IN = results['sigmas_in']
params = results['params']
Y = results['y']

# evaluation_params = {
#     'var_names': ['DGS3', 'inf', 'unrate'],
#     'is_test': False,
#     'exps_to_plot': [0],
#     'test_size': 40
# }

# Evaluation.evaluate_TVPs(results, benchmark_results = None, evaluation_params = evaluation_params, image_folder_path = image_folder_path)

# Need to undo constants later
irf_params = {
    'n_var': 3,
    'n_lags': 1,
    'n_betas': 4,
    'max_h': 10,
    'var_names': ['DGS3', 'inf', 'unrate']
}

# Sum betas across all hemispheres
BETAS_IN_all_hemi = np.sum(BETAS_IN, axis = -1)

IRF = IRFConditional(irf_params)

IRFS_median = IRF.get_irfs(BETAS_IN_all_hemi, SIGMAS_IN)

IRF.plot_irfs(IRFS_median, image_folder_path)

print('IRFS_median', IRFS_median.shape)