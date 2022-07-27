from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import os

class Benchmarks:

  def __init__(self, dataset, benchmark_params, run_name):

    self.run_name = run_name
    self.folder_path = f'results/{run_name}'
    # Create benchmark folder if not exist yet
    self.benchmark_folder_path = f'{self.folder_path}/benchmarks'
    if os.path.isdir(self.benchmark_folder_path) == False:
      os.mkdir(self.benchmark_folder_path)

    # Subset the dataset to only the variables desired
    self.var_names = benchmark_params['var_names']
    self.n_var = len(self.var_names)
    self.dataset = dataset[self.var_names]

    self.n_lag_linear = benchmark_params['n_lag_linear']
    self.n_lag_d = benchmark_params['n_lag_d']
    self.benchmarks = benchmark_params['benchmarks']
    
    self.test_size = benchmark_params['test_size']

    self.window_length = benchmark_params['window_length']
    self.reestimation_window = benchmark_params['reestimation_window']

    # Process the dataset
    self.X_train, self.Y_train, self.X_test, self.Y_test, self.x_pos = self._process_dataset(self.dataset)

  def _process_dataset(self, dataset):
    
    mat_data_d = dataset
    # 2: Generating the lags
    for col in self.var_names:
      for lag in range(1, self.n_lag_linear + 1):
        mat_data_d[f'{col}.l{lag}'] = mat_data_d[col].shift(lag)

    mat_data_d = mat_data_d.iloc[self.n_lag_d:, :].reset_index(drop=True)
    mat_y_d = mat_data_d.iloc[:, :self.n_var]
    mat_x_d = mat_data_d.iloc[:, self.n_var:]
    mat_x_d_colnames = mat_data_d.iloc[:, self.n_var:].columns

    # mat_x_d and mat_y_d have n_lag_d fewer observations than original data
    n_obs = mat_y_d.shape[0]
    train_split_id = n_obs - self.test_size

    # Get the index of the lagged values of unemployment rate
    first_parts = ['.l' + str(lag) for lag in range(1, self.n_lag_linear + 1)]

    get_xpos = lambda variable_name, first_parts: [list(i for i, n in enumerate(mat_x_d_colnames) if n == variable_name + first_part)[0] for first_part in first_parts]

    x_pos = {}
    for var in self.var_names:
      x_pos[var] = get_xpos(var, first_parts)

    X_train = np.array(mat_x_d.iloc[:train_split_id, :])
    Y_train = np.array(mat_y_d.iloc[:train_split_id, :])
    X_test = np.array(mat_x_d.iloc[train_split_id:, :])
    Y_test = np.array(mat_y_d.iloc[train_split_id:, :])

    return X_train, Y_train, X_test, Y_test, x_pos

  # Wrapper function to conduct VAR and AR estimation
  def _conduct_regression(self, X_train, X_test, Y_train, Y_test, var = True):

    # Do the VAR
    if var == True:
      # Fit the linear regression
      lin_reg = LinearRegression(fit_intercept = True)
      fit = lin_reg.fit(X_train, Y_train)
      # Make in-sample predictions
      preds_train = lin_reg.predict(X_train)
      preds_test = lin_reg.predict(X_test)

      # Return coefs
      coefs = lin_reg.coef_
      intercept = np.expand_dims(lin_reg.intercept_, axis = 1)
      coefs = np.concatenate((intercept, coefs), axis = 1)

    else:
      # Equation-by-equation AR model
      preds_train = np.zeros_like(Y_train)
      preds_test = np.zeros_like(Y_test)

      coefs = np.zeros((self.n_var, self.n_var * self.n_lag_linear + 1))
      
      for i in range(self.n_var):
        var_name = self.var_names[i]
        y_train_var = Y_train[:, i]
        X_train_var = X_train[:, self.x_pos[var_name]]

        lin_reg = LinearRegression(fit_intercept = True)
        fit = lin_reg.fit(X_train_var, y_train_var)

        X_test_var = X_test[:, self.x_pos[var_name]]
        preds_train[:, i] = lin_reg.predict(X_train_var)
        preds_test[:, i] = lin_reg.predict(X_test_var)

        coefs[i, self.x_pos[var_name]] = lin_reg.coef_
    #      coefs[i, (n_lag_linear*i+1):(n_lag_linear*(i+1)+1)] = lin_reg.coef_
        coefs[i, 0] = lin_reg.intercept_

    return preds_train, preds_test, coefs

  # @title Rolling & Expanding window regression
  def _conduct_window_regression(self, X_train, X_test, Y_train, Y_test, window_length = 40, reestimation_window = 1, window_type = 'roll', var = True):

    # Note: For rolling window: in-sample preds are stored in preds_train_in_all acc to their start id. 
    # For expanding window: end-id

    # In-sample preds for the train set: n_obs x n_var x n_obs (start_t)
    preds_train_in_all = np.zeros((Y_train.shape[0], Y_train.shape[1], Y_train.shape[0]))
    preds_train_in_all[:] = np.nan
    # Out-of-sample preds for the train set (for every window, 'reestimation_window' preds will be predicted)
    preds_train_out = np.zeros_like(Y_train)
    preds_train_out[:] = np.nan
    # Preds for the test set 
    preds_test = np.zeros_like(Y_test)
    preds_test[:] = np.nan

    # Betas for the train set
    betas_all = np.zeros((Y_train.shape[0], Y_train.shape[1], (Y_train.shape[1] * self.n_lag_linear + 1), Y_train.shape[0]))
    betas_all[:] = np.nan

    for start_t in range(0, Y_train.shape[0] - window_length + 1, reestimation_window):

      end_t = start_t + window_length - 1
      if window_type == 'roll':
        X_train_subset = X_train[start_t:(start_t + window_length), :]
        Y_train_subset = Y_train[start_t:(start_t + window_length), :]
      elif window_type == 'expand':
        X_train_subset = X_train[:(start_t + window_length), :]
        Y_train_subset = Y_train[:(start_t + window_length), :]
      X_train_out_subset = X_train[(start_t + window_length):(start_t + window_length + reestimation_window), :]
      Y_train_out_subset = Y_train[(start_t + window_length):(start_t + window_length + reestimation_window), :]

      # Estimate the model using this window of data
      if start_t == Y_train.shape[0] - window_length:
        preds_train_in_subset, preds_train_out_subset, coefs = self._conduct_regression(X_train_subset, X_train_subset, Y_train_subset, Y_train_subset, var = var)
      else:
        preds_train_in_subset, preds_train_out_subset, coefs = self._conduct_regression(X_train_subset, X_train_out_subset, Y_train_subset, Y_train_out_subset, var = var)
        preds_train_out[(start_t + window_length):(start_t + window_length + reestimation_window), :] = preds_train_out_subset
      
      if window_type == 'roll':
        preds_train_in_all[start_t:(start_t + window_length) , :, start_t] = preds_train_in_subset
        betas_all[start_t:(start_t + window_length), :, :, start_t] = coefs
      elif window_type == 'expand':
        preds_train_in_all[:(start_t + window_length) , :, end_t] = preds_train_in_subset
        betas_all[:(start_t + window_length), :, :, end_t] = coefs
      
    # Get the test set preds
    _, preds_test_subset, _ = self._conduct_regression(X_train_subset, X_test, Y_train_subset, Y_test, var = var)
    preds_test[:] = preds_test_subset[:]

    # Get the in-sample train preds by taking the average of all
    preds_train_in = np.nanmean(preds_train_in_all, axis = 2)

    # Get the Time-varying Coefs by taking average
    betas = np.nanmean(betas_all, axis = 3)

    return preds_train_in, preds_train_out, preds_test, betas

  def _conduct_random_forest(self, X_train, X_test, Y_train, Y_test):

    preds_train = np.zeros_like(Y_train)
    preds_test = np.zeros_like(Y_test)
    
    # Estimate the RF model equation by equation
    for i in range(self.n_var):
      print(f'Computing RF Benchmark for variable {i}, time: {datetime.now()}')
      var_name = self.var_names[i]
      y_train_var = Y_train[:, i]
      X_train_var = X_train[:, self.x_pos[var_name]]

      rf = RandomForestRegressor(max_depth = 5, random_state = 42, n_estimators = 100, n_jobs = None)
      fit = rf.fit(X_train_var, y_train_var)

      X_test_var = X_test[:, self.x_pos[var_name]]
      preds_train[:, i] = rf.predict(X_train_var)
      preds_test[:, i] = rf.predict(X_test_var)
    
    return preds_train, preds_test, None


  # Wrapper Function - public
  def compute_benchmarks(self):

    X_train, X_test, Y_train, Y_test = self.X_train, self.X_test, self.Y_train, self.Y_test
    window_length = self.window_length
    reestimation_window = self.reestimation_window

    # Entire window regression 
    preds_train_whole_var, preds_test_whole_var, betas_whole_var = self._conduct_regression(X_train, X_test, Y_train, Y_test, var = True)
    preds_train_whole_ar, preds_test_whole_ar, betas_whole_ar = self._conduct_regression(X_train, X_test, Y_train, Y_test, var = False)

    # Rolling window regression
    preds_train_in_roll_var, _, preds_test_roll_var, betas_roll_var = self._conduct_window_regression(X_train, X_test, Y_train, Y_test, window_length = window_length, reestimation_window = reestimation_window, window_type = 'roll', var = True)
    preds_train_in_roll_ar, _, preds_test_roll_ar, betas_roll_ar = self._conduct_window_regression(X_train, X_test, Y_train, Y_test, window_length = window_length, reestimation_window = reestimation_window, window_type = 'roll', var = False)

    # Expanding window regression
    preds_train_in_expand_var, _, preds_test_expand_var, betas_expand_var = self._conduct_window_regression(X_train, X_test, Y_train, Y_test, window_length = window_length, reestimation_window = reestimation_window, window_type = 'expand', var = True)
    preds_train_in_expand_ar, _, preds_test_expand_ar, betas_expand_ar = self._conduct_window_regression(X_train, X_test, Y_train, Y_test, window_length = window_length, reestimation_window = reestimation_window, window_type = 'expand', var = False)

    # Random forest
    preds_train_whole_rf, preds_test_whole_rf, _ = self._conduct_random_forest(X_train, X_test, Y_train, Y_test)

    # Naive benchmarks
    preds_train_zero, preds_test_zero = get_naive_preds(Y_train, Y_test, method = 'zero')
    preds_train_mean, preds_test_mean = get_naive_preds(Y_train, Y_test, method = 'mean')

    # Expanding the entire-window regression coefs to be same shape as the rolling/expanding window
    betas_whole_var = np.repeat(np.expand_dims(betas_whole_var, axis = 0), betas_expand_var.shape[0], axis = 0)
    betas_whole_ar = np.repeat(np.expand_dims(betas_whole_ar, axis = 0), betas_expand_ar.shape[0], axis = 0)

    savefile_header = ''

    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_VAR_whole.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_whole_var, betas_in = betas_whole_var, test_preds = preds_test_whole_var, y = Y_train, y_test = Y_test)
    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_AR_whole.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_whole_ar, betas_in = betas_whole_ar, test_preds = preds_test_whole_ar, y = Y_train, y_test = Y_test)
    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_VAR_roll.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_in_roll_var, betas_in = betas_roll_var, test_preds = preds_test_roll_var, y = Y_train, y_test = Y_test)
    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_AR_roll.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_in_roll_ar, betas_in = betas_roll_ar, test_preds = preds_test_roll_ar, y = Y_train, y_test = Y_test)
    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_VAR_expand.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_in_expand_var, betas_in = betas_expand_var, test_preds = preds_test_expand_var, y = Y_train, y_test = Y_test)
    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_AR_expand.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_in_expand_ar, betas_in = betas_expand_ar, test_preds = preds_test_expand_ar, y = Y_train, y_test = Y_test)
    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_RF_whole.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_whole_rf, test_preds = preds_test_whole_rf, y = Y_train, y_test = Y_test)
    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_zero.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_zero, test_preds = preds_test_zero, y = Y_train, y_test = Y_test)
    with open(f'{self.benchmark_folder_path}/{savefile_header}benchmark_mean.npz', 'wb') as f:
      np.savez(f, train_preds = preds_train_mean, test_preds = preds_test_mean, y = Y_train, y_test = Y_test)

def get_naive_preds(Y_train, Y_test, method = 'zero'):
  '''
  Types of preds:
  zero - zero
  mean - mean of the variable in training set
  median - median of the variable in training set
  '''
  if method == 'zero':
    return np.zeros_like(Y_train), np.zeros_like(Y_test)
  elif method == 'mean':
    return (
        np.repeat(np.expand_dims(np.mean(Y_train, axis = 0), axis = 0), Y_train.shape[0], axis = 0),
        np.repeat(np.expand_dims(np.mean(Y_train, axis = 0), axis = 0), Y_test.shape[0], axis = 0)
    )
  elif method == 'median':
    return (
        np.repeat(np.expand_dims(np.median(Y_train, axis = 0), axis = 0), Y_train.shape[0], axis = 0),
        np.repeat(np.expand_dims(np.median(Y_train, axis = 0), axis = 0), Y_test.shape[0], axis = 0)
    )