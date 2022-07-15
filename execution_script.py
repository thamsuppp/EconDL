import torch
import sys

from EconDL.Evaluation import Evaluation
from EconDL.Run import Run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment name is the command-line argument
run_name = sys.argv[1]

# Instantiate the Run:
# - Creates a new folder
# - Reads the run's configuration json file
# - Instantiates the experiments with the corresponding nn_hyps
# - Loads the data into the RunObj
RunObj = Run(run_name, device)

# Train all experiments within the run and store the experiments within the object
RunObj.train_all()

# Runs evaluation for the trained run
EvaluationObj = Evaluation(RunObj)
print(EvaluationObj.check_results_sizes())
EvaluationObj.plot_all()
