import torch
import sys
import random
import os
import json
import time
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device', device)

# Experiment name is the command-line argument
run_name = sys.argv[1]
num_repeats = int(sys.argv[2])

folder_path = f'results/{run_name}'
if os.path.isdir(folder_path) == False:
  # Wait for a random number of seconds to avoid collision
  time.sleep(random.randint(0, 10))
  os.mkdir(folder_path)
  print(f'Folder made at {folder_path}')
else:
  print(f'Folder at {folder_path} already exists')

instance_id = int(os.environ.get('SLURM_ARRAY_TASK_ID')) - 1

# Logic that calculates the number of experiment instances to instantiate
with open(f'exp_config/{run_name}.json', 'r') as f:
  all_params = json.load(f)
num_experiments = len(all_params['nn_hyps'])

test_size = all_params['run_params']['test_size']

num_instances = 0
instance_exp_mapping = {}

for exp in range(num_experiments):
  if all_params['nn_hyps'][exp]['reestim_params']['reestim'] == False:
    for repeat in range(num_repeats):
      instance_exp_mapping[num_instances] = {'exp': exp, 'reestim': 0, 'repeat': repeat}
      num_instances += 1

  else:
    num_reestims = int(test_size / all_params['nn_hyps'][exp]['reestim_params']['reestimation_window'])
    for reestim in range(num_reestims):
      for repeat in range(num_repeats):
        instance_exp_mapping[num_instances] = {'exp': exp, 'reestim': reestim, 'repeat': repeat}
        num_instances += 1

print(f'# Experiments: {num_experiments}, # Instances: {num_instances}')
print(f'Instance to experiment mapping: {instance_exp_mapping}')


# Run the experiment that is specified by this instance
experiment_id = instance_exp_mapping[instance_id]['exp']
reestim = instance_exp_mapping[instance_id]['reestim']
repeat = instance_exp_mapping[instance_id]['repeat']

print(f'Instance {instance_id}, {instance_exp_mapping[instance_id]}')
RunObj = Run(run_name, device, experiment_id = experiment_id, job_id = repeat, reestim_id = reestim)
RunObj.train_all()

if experiment_id == 0: # Oly train the ML experiments once per across all experiments (only needed once)
  RunObj.train_ml_experiments()


'''
Note: 
- To run a parallel run, set job_id to the SLURM ID
- If we are evaluating results from a parallel run, set job_id to None, comment out train_all, and uncomment compile_experiments()
'''