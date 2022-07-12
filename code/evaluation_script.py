import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from IRF.IRFConditional import IRFConditional
from Evaluation import Evaluation


# Experiment name is the command-line argument
experiment_name = sys.argv[1]
folder_path = f'../results/{experiment_name}'

evaluation_params = {
    'var_names': ['DGS3', 'inf', 'unrate'],
    'is_test': False,
    'test_size': 100,
    'need_to_combine': False,
    'is_simulation': False,
    'multiple_datasets': False,
    'sim_dataset': 2,
    'benchmarks': ['VAR_whole', 'AR_whole', 'VAR_roll', 'AR_roll', 'VAR_expand', 'AR_expand', 'RF_whole'],
    'num_bootstraps': 25,
    'test_size': 100,
    'M_varnn': 1,
    'experiments_to_load': [0],
    'plot_all_bootstraps': False,
    'exps_to_plot': [0]
}

irf_params = {
    'n_var': 3,
    'n_lags': 1,
    'n_betas': 4,
    'max_h': 10,
    'var_names': ['DGS3', 'inf', 'unrate']
}

# Create an Evaluation instance
EvaluationObj = Evaluation(experiment_name, evaluation_params)
print(EvaluationObj.BETAS_ALL.shape)
print(EvaluationObj.check_results_sizes())

EvaluationObj.plot_predictions()
EvaluationObj.plot_errors(data_sample='oob')
EvaluationObj.plot_errors(data_sample='test', exclude_last=20)

# # Plot the TVPs
# EvaluationObj.evaluate_TVPs()
# EvaluationObj.evaluate_cholesky()
# EvaluationObj.evaluate_precision()
# EvaluationObj.evaluate_sigmas()



# # Sum betas across all hemispheres
# BETAS_IN_all_hemi = np.sum(BETAS_IN, axis = -1)

# IRF = IRFConditional(irf_params)

# IRFS_median = IRF.get_irfs(BETAS_IN_all_hemi, SIGMAS_IN)

# IRF.plot_irfs(IRFS_median, image_folder_path)

# print('IRFS_median', IRFS_median.shape)