# EconDL v1.1 (20 Jul 2022)

## Instructions:

- Change your experiment parameters by creating a new json file like `{run_name}.json` in the `exp_config` folder. Make sure not to leave out any parameters (look at the example, `18jul_test.json`)
    - For nn_hyps that don't change throughout experiments, they are defined in the `nn_hyps_default.json` file or wherever you set the `default_nn_hyps` parameter to be. You can also throw those parametes into `run_params` (e.g. for `18jul_test.json`, all experiments have the same `constant_tvpl`, so I put them in `run_params`)
    - `s_pos`: It is a list of lists, with each inner list having 2 numbers - start and end_id of the variables you want in that hemisphere.
        - For now, we have set it such that only endogenous lags enter the VARNN. This is because unconditional IRF and multi-horizon forecasting can't use exogenous time lags for now. (I am currently working on an update that will allow you to train 2-hemisphere VARNNs, and then train a separate endogenous-only VARNN for the unconditional IRF and multi-horizon forecasting)
        - Make sure `s_pos` adds up (there should be (n_lag_linear + n_lag_d) * n_var inputs into the NN). If you put a higher number in `s_pos` it will lead to a CUDA error.
- To run the experiment, run `python execution_script.py {run_name} {num_experiments}` in command-line. (On Compute Canada, put this command in the .sh file and run `sbatch 18jul_test.sh` as before)
    - It is important that `run_experiments` is correct i.e. the number of elements in the `nn_hyps` error in your experiment json file. This is so that the code can correctly divide the Compute Canada cores to calculate different repeats and experiments properly. 
        - e.g. If there are 4 experiments and I want to run 3 repeats per experiment, I will specify 1-12 in the Compute Canada .sh file. This means that cores 1, 5, 9 will run experiment 0, cores 2,6,10 will run experiment 1, ...
- Evaluation: When all the cores are done, run `python evaluation_script.py {run_name}` in command-line (or via .sh file in Compute Canada) to run the Evaluation object which plots all the graphs.
    - If some repeats failed and you want to exclude them from the evaluation, you can use the `repeats_to_include` parameter in the json file to specify the repeats you want to include.

## Parameters to Note

- `constant_tvpl` - number of neurons in the TVPL layer for the constant. Must be wrapped by a list i.e. [40] (added 7/18)

## Sequence of Events

**Training (execution_script.py)**

- For each run:
    - train_experiments() - For each experiment
        - For each repeat:
            - train_experiment()
            - compute_unconditional_irf()
            - compute_multi_forecasting()
    - train_multi_forecasting_benchmarks()
    - train_benchmarks()
    - For each ML experiment:
        - For each repeat:
            - train_experiment()
            - compute_unconditional_irf()
            - compute_multi_forecasting()

**Evaluation (evaluation_script.py)**

- For each experiment:
    - load_results - loads uncompiled results into `self.results_uncompiled` (only repeats specified in ‘repeats_to_include)
    - compile_results:
        - For VARNN Experiments: results, unconditional IRF **- and plots it**, multi forecasting
        - For ML Experiments: unconditional IRF **- and plots it**, multi forecasting
    - compute_conditional_irf (needs compiled results) - computes and **plots conditional IRF results**
    - plot_all() - plots VAR IRFs, cholesky, precision, sigmas, betas, one-step predictions, OOB and test errors, multi-step forecasts


## Version History

- v1.1 (7/20/22): Self-tuning ML Benchmarks, Running experiments w multiple hemispheres
- v1.0 (7/19/22): Experiment and Evaluation code works - verified with 18 Jul example.
- What doesn't work/not implemented: prior shift, simulations evaluation, VSN 

## File Structure

- Run
    - Experiment
        - Repeat
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