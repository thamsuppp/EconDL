import torch
import sys
import os
import argparse
from EconDL.Run import Run
from EconDL.Evaluation import Evaluation

parser = argparse.ArgumentParser()
parser.add_argument('run_name', help = 'Name of run to evaluate')
parser.add_argument('--no_cond', help = 'Turn off conditional IRFs', action = 'store_true')
parser.add_argument('--fcast_only', help = 'Only plot forecasts', action = 'store_true')

cond_irf = True
fcast_only = False
args = parser.parse_args()
run_name = args.run_name
if args.no_cond:
  cond_irf = False
if args.fcast_only:
  fcast_only = True

print(f'Run is {run_name}, cond_irf is {cond_irf}, fcast_only is {fcast_only}')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RunObj = Run(run_name, device)

# Train benchmarks if not already trained (function in Run will not train if benchmark files already exist)
RunObj.train_benchmarks()
RunObj.train_multi_forecast_benchmarks()

EvaluationObj = Evaluation(RunObj)
if cond_irf == True:
  EvaluationObj.Run.compute_conditional_irfs()

print(EvaluationObj.check_results_sizes())
if fcast_only == True:
  EvaluationObj.plot_forecasts()
else:
  EvaluationObj.plot_all()