import torch
import sys
import os
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation

run_name = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device', device)
# If we are doing this in parallel, then we pass in the job_id parameter here
RunObj = Run(run_name, device)

'''
Instantiating Evaluation does this:
- compiles results for all experiment's repeats (VARNN + ML)
- loads results
'''
EvaluationObj = Evaluation(RunObj)
EvaluationObj.Run.compute_conditional_irfs()

print(EvaluationObj.check_results_sizes())
EvaluationObj.plot_all()


# Run obj runs the benchmarks

# Evaluation object needs to compile all the results (bascially the 'compile_results' should be the responsibility of the Evaluation object now instead of Experiment
# Evaluation object should also run the conditional_irf - stuff that 


# print(EvaluationObj.check_results_sizes())

