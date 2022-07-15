import torch
import sys
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment name is the command-line argument
run_name = sys.argv[1]

# Instantiate the Run:
# - Creates a new folder
# - Reads the run's configuration json file
# - Instantiates the experiments with the corresponding nn_hyps
# - Loads the data into the RunObj

# If we are doing this in parallel, then we pass in the job_id parameter here
RunObj = Run(run_name, device, job_id = None)

# Train all experiments within the run and store the experiments within the object
RunObj.train_all()
#RunObj.compile_experiments()

# Runs evaluation for the trained run
EvaluationObj = Evaluation(RunObj)
print(EvaluationObj.check_results_sizes())
EvaluationObj.plot_all()

'''
Note: 
- To run a parallel run, set job_id to the SLURM ID
- If we are evaluating results from a parallel run, set job_id to None, comment out train_all, and uncomment compile_experiments()
'''