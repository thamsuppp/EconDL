import json
import torch
import sys
import os
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation

# Experiment name is the command-line argument
run_name = sys.argv[1]

print(os.getcwd())

with open(f'exp_config/pre-processed/{run_name}.json', 'r') as f:
  all_params = json.load(f)

# If we just want to make 1 incremental change to the baseline model
baseline_params = all_params['nn_hyps_baseline']
params_to_vary = all_params['params_to_vary']

nn_hyps = [baseline_params]

experiment_id = 1
experiment_groups = {}
for param, param_values in params_to_vary.items():
  experiment_group_ids = [0]
  for param_value in param_values:
    baseline_params_copy = baseline_params.copy()
    baseline_params_copy.update({param: param_value, 'name': f'{param} {param_value}'})
    experiment_group_ids.append(experiment_id)
    experiment_id += 1
    nn_hyps.append(baseline_params_copy)

  experiment_groups.update({param: experiment_group_ids})

print(baseline_params)
print(nn_hyps)
print('Length', len(nn_hyps))

all_params['experiment_groups'] = experiment_groups
all_params['nn_hyps'] = nn_hyps

# Save the new JSON file
with open(f'exp_config/{run_name}.json', 'w') as f:
  json.dump(all_params, f, indent = 4)