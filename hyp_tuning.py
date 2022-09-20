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

# Load hyperparams file
hyperparams_file_path = f'exp_config/hyperparams/{run_name}.txt'

# Read the entire txt file
with open(hyperparams_file_path, 'r') as f:
  hyperparams = f.read()

# Execute the hyperparams file
exec(hyperparams)

BayesianHypTuningObj = BayesianHypTuning(run_name, device, dimensions = dimensions, dim_names = dim_names, default_parameters = default_parameters)
BayesianHypTuningObj.conduct_hyp_tuning()