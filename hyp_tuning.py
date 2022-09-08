import torch
import sys
import os
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

from EconDL.Run import Run
from EconDL.BayesianHypTuning import BayesianHypTuning



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device', device)

# Experiment name is the command-line argument
run_name = sys.argv[1]

folder_path = f'results/{run_name}'
if os.path.isdir(folder_path) == False:
  os.mkdir(folder_path)
  print(f'Folder made at {folder_path}')
else:
  print(f'Folder at {folder_path} already exists')

 # List of hyperparameters to tune
dim_dropout_rate = Real(low = 0, high = 0.75, prior = 'uniform', name = 'dropout_rate')
dim_nn_width = Integer(low = 10, high = 400, prior = 'log-uniform', name = 'nn_width')
dim_nn_depth = Integer(low = 2, high = 6, prior = 'uniform', name = 'nn_depth')
dim_tvpl_width = Integer(low = 2, high = 25, prior = 'log-uniform', name = 'tvpl')
dim_constant_tvpl = Integer(low = 10, high = 200, prior = 'log-uniform', name = 'constant_tvpl')
dim_precision_lambda = Real(low = 0.00, high = 0.25, prior = 'uniform', name = 'precision_lambda')
dim_lr = Real(low = 0.00001, high = 0.01, prior = 'log-uniform', name = 'lr')
dim_activation = Categorical(categories = ['nn.ReLU()', 'nn.SELU()'], name = 'activation')

dimensions = [dim_dropout_rate, dim_nn_width, dim_nn_depth, dim_tvpl_width, dim_constant_tvpl, dim_precision_lambda, dim_lr, dim_activation]
dim_names = ['dropout_rate', 'nn_width', 'nn_depth', 'tvpl', 'constant_tvpl', 'precision_lambda', 'lr', 'activation']
default_parameters = [0.0001, 200, 3, 5, 100, 0.025, 0.0001, 'nn.SELU()']

BayesianHypTuningObj = BayesianHypTuning(run_name, device, dimensions = dimensions, dim_names = dim_names, default_parameters = default_parameters)
BayesianHypTuningObj.conduct_hyp_tuning()