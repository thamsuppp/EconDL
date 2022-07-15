# EconDL



## Instructions:

- Change your experiment parameters by creating a new json file like `{run_name}.json` in the `exp_config` folder. Make sure not to leave out anything (look at the example, `14jul_test.json`)
    - For nn_hyps that don't change throughout experiments, they are defined in the `nn_hyps_default.json` file or wherever you set the `default_nn_hyps` parameter to be.
- To run the experiment, run `python execution_script_13jul.py {run_name}` in command-line. 
    This will instantiate a Run object


## File Structure

- Run
    - Experiment
        - Repeats
- Classes:
    - Run (run-level)
    - Experiment (experiment-level)
    - Evaluation (run-level)
    - Benchmarks (run-level)
    - ForecastBenchmarks (run-level)
    - IRFConditional (experiment-level)
    - IRFUnconditional (repeat-level)
    - IRFUnconditionalEvaluation (run-level)
    - ForecastMulti (repeat-level)