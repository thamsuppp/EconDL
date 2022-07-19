import torch
import sys
import os
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation

run_name = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RunObj = Run(run_name, device)

EvaluationObj = Evaluation(RunObj)
EvaluationObj.Run.compute_conditional_irfs()

print(EvaluationObj.check_results_sizes())
EvaluationObj.plot_all()
