num_inner_bootstraps = 100
num_repeats = 2


nn_hyps = {
    # Mostly unchanged hyperparameters
    'epochs': 500,
    'show_train': 3,
    'opt_bootstrap': 2,
    'num_bootstrap': num_inner_bootstraps, 
    'sampling_rate': 0.75,
    'block_size': 12,
    'cancel_out': False,
    'standardize': True,
    'prior_shift': False,
    'prior_shift_strength': 0, 
    'oob_loss_multiple_threshold': 5,
    'save_models': False,
    'exog': None,
    'tol': 0.0001,


    # Hyperparamters of interest - but not changed in this experiment
    'nodes': [400, 200, 100],
    'tvpl_archi': [5],
    'patience': 50,
    'lr': 0.0005,
    'lr_multiple': 0.9975,
    'dropout_rate': 0.25,
    'actv': 'nn.ReLU()',

    'time_dummy_setting': 2, 
    'marx': True,
    
    'dummy_interval': 6,
    'l1_input_lambda': 0,
    'l0_input_lambda': 0,
    'precision_lambda': 0,
    'input_dropout_rate': 0,
    'vsn': False, 
    'fcn': False, 
    'eqn_by_eqn': False,
    'time_hemi_prior_variance': 1,
    'fix_bootstrap': False,
    'loss_weight_param': 0.5,
    
    'log_det_multiple': 1,
    'precision_lambda': 0.05, 
    'lambda_temper_epochs': 25,
    'optimizer': 'Adam',
    'n_lag_d': 12, 'n_lag_linear': 6, 'n_lag_ps': 6,
    'variables': ['DGS3', 'inf', 'unrate'],

    's_pos_setting': {'is_hemi': True, 'n_times': 123}, 'joint_estimation': True, 
    's_pos': [list(range(54)), list(range(54, 173))], 
    'name': 'Joint RELU New',
    'test_size': 100

}