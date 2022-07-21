import torch
import sys
import os
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device', device)

# Experiment name is the command-line argument
run_name = sys.argv[1]
num_experiments = int(sys.argv[2])
num_repeats = int(sys.argv[3])

folder_path = f'results/{run_name}'
if os.path.isdir(folder_path) == False:
  os.mkdir(folder_path)
  print(f'Folder made at {folder_path}')
else:
  print(f'Folder at {folder_path} already exists')

# Instantiate the Run:
# - Creates a new folder
# - Reads the run's configuration json file
# - Instantiates the experiments with the corresponding nn_hyps
# - Loads the data into the RunObj

for repeat in range(num_repeats):
  for experiment_id in range(num_experiments):

    print(f'Experiment {experiment_id}, repeat {repeat}')
    # Train the specific experiment_id and speciifc repeat_id
    RunObj = Run(run_name, device, experiment_id = experiment_id, job_id = repeat)
    RunObj.train_all()

    if experiment_id == 0: # Oly train the ML experiments once per across all experiments (only needed once)
      RunObj.train_ml_experiments()


EvaluationObj = Evaluation(RunObj)
print(EvaluationObj.check_results_sizes())
EvaluationObj.plot_cholesky()
EvaluationObj.plot_precision()
EvaluationObj.plot_sigmas()
EvaluationObj.plot_betas()
EvaluationObj.plot_predictions()
EvaluationObj.plot_errors(data_sample='oob')
EvaluationObj.plot_errors(data_sample='test')
