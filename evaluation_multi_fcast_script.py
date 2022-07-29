import torch
import sys
import os
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('run_name', help = 'Name of run to evaluate')

cond_irf = True
fcast_only = False
args = parser.parse_args()
run_name = args.run_name

RunObj = Run(run_name, device)
EvaluationObj = Evaluation(RunObj)

print(EvaluationObj.check_results_sizes())
EvaluationObj.evaluate_multi_step_forecasts()