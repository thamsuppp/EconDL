import json
import torch
import sys
import os
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation

# Experiment name is the command-line argument
run_name = sys.argv[1]

print(os.getcwd())

with open(f'exp_config/reestim/{run_name}.json', 'r') as f:
  all_params = json.load(f)

# If we just want to make 1 incremental change to the baseline model
baseline_params = all_params['nn_hyps_baseline']

nn_hyps = []

max_test_size = all_params['run_params']['reestim_params']['max_test_size']
reestimation_window = all_params['run_params']['reestim_params']['reestimation_window']
num_reestimations = int(max_test_size / reestimation_window)

for r in range(num_reestimations):

  # Test size
  test_size = (num_reestimations - r) * reestimation_window

  baseline_params_copy = baseline_params.copy()
  baseline_params_copy.update({'name': f'Reestimation {r}', 'test_size': test_size})
  nn_hyps.append(baseline_params_copy)

print(baseline_params)
print('Length', len(nn_hyps))

all_params['nn_hyps'] = nn_hyps

# Save the new JSON file
with open(f'exp_config/{run_name}.json', 'w') as f:
  json.dump(all_params, f, indent = 4)