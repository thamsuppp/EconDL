from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random

from utils import invert_scaling
from predict_nn import predict_nn_new, predict_nn_old


class IRFUnconditional:

  def __init__(self, experiment_name, irf_params, device = None):

    self.experiment_name = experiment_name
    self.n_lag_linear = irf_params['n_lag_linear']
    self.n_lag_d = irf_params['n_lag_d']
    self.n_var = irf_params['n_var']
    self.max_h = irf_params['max_h']
    self.var_names = irf_params['var_names']
    
    self.num_simulations = irf_params['num_simulations']
    self.start_shock_time = irf_params['start_shock_time']
    self.endh = irf_params['endh']
    self.forecast_method = irf_params['forecast_method']
    
    self.plot_all_bootstraps = irf_params['plot_all_bootstraps']

    self.device = device

    # Simulation time steps that the impulses are done
    self.randoms = list(range(self.start_shock_time, self.num_simulations, self.endh))

    # Results
    self.FCAST = None 
    self.FCAST_COV_MAT = None
    self.irf_mat = None
    self.irf_cum_mat = None
  


  # @title New IRF Simulation Wrapper Function (Joint)
  # Returns: fcast, fcast_cov_mat, sim_shocks
  ## kk: variable to shock 
  # k: response of shock
  # Wrapper function that: 1) takes residuals from model predictions
  # 2) Generates shocks, 3) Simulates path given shocks at specific random indices
  # Returns fcast - 2000 time steps x 5 shock variables x 5 response variables x 2 shocks
  def simulate_irf_paths_new(self, Y_train, Y_test, results, device):
    n_var = self.n_var

    # Get the residuals - resiudals are the average of the different bootstrap iterations
    oos_res = Y_test - results['pred']
    oob_res_save = Y_train - results['pred_in']
    oob_res = oob_res_save[~np.isnan(oob_res_save).any(axis=1)]
    oob_res = np.vstack([oob_res, oos_res])

    # Time periods where we create the impulses
    impulse_times = list(range(self.start_shock_time, self.num_simulations, self.endh))
    impulse_times_all = []

    # Shock the system in plausible way (shock few days in a row)
    for e in impulse_times:
      for sj in [0]:
        impulse_times_all.append(e + sj)

    fcast = np.zeros((self.num_simulations, n_var, n_var, 3))
    fcast[:] = np.nan

    fcast_cov_mat = np.zeros((self.num_simulations, n_var, n_var, n_var, 3))
    fcast_cov_mat[:] = np.nan

    # sim_shocks: self.num_simulations x n_var
    sim_shocks = np.random.multivariate_normal([0] * n_var, np.eye(n_var), size = self.num_simulations)
    for simul in range(self.num_simulations):
      sim_shocks[simul, :] = np.random.multivariate_normal([0] * n_var, np.eye(n_var), size = 1)

    for kk in range(n_var): # Variable to shock
      for shock_level in [0, 1]:

        # Initialize new data at 0
        new_in_linear = np.zeros(self.n_lag_linear * n_var)
        new_in_nonlinear = np.zeros(self.n_lag_d * n_var)
        fcast[0, :, kk, shock_level] = np.zeros((n_var))

        # Start the simulation
        f = 1
        while f < self.num_simulations:
          ### 1: Construct input
          if f % 100 == 0:
            print(f, datetime.now())

          # Add the newly observed data to the model (new variables become the L0, appended to front,
          # while we drop the most lagged variables)
          new_in_linear = np.hstack([fcast[f-1, :, kk, shock_level], new_in_linear[:(len(new_in_linear) - n_var)]])
          new_in_nonlinear = np.hstack([fcast[f-1, :, kk, shock_level], new_in_nonlinear[:(len(new_in_nonlinear) - n_var)]])
          
          # Generate MARX transformed variables - for that one new day
          new_data_marx = new_in_nonlinear.copy()
          for lag in range(2, self.n_lag_d + 1):
            for var in range(n_var):
              who_to_avg = list(range(var, n_var * (lag - 1) + var + 1, n_var))
              new_data_marx[who_to_avg[-1]] = new_in_nonlinear[who_to_avg].mean()

          # Combine the first n_lag_linear lags, with the MARX data, to get the full 325-dim input vector
          new_data_all = np.hstack([new_in_linear, new_data_marx])
          new_data_all = np.expand_dims(new_data_all, axis = 0)

          ### 2: Call NN forward to get pred and cov mat (stop if the whole thing exploded)
          if np.any(np.isnan(new_data_all)) == False and np.all(np.isfinite(new_data_all)) == True:
            #print(f'Simul Timestep {f}')
            pred, cov = self.predict_nn_new(results, new_data_all, device)
            #print(f'Pred: {pred}, Cov: {cov}')

            # Cholesky the cov mat to get C matrix
            cov = np.squeeze(cov, axis = 0)
            c_t = np.linalg.cholesky(cov)

            fcast_cov_mat[f, :, :, kk, shock_level] = cov

            ### 3: Sample 1 shock from normal distribution
            sim_shock = sim_shocks[f, :].copy()

            # @DEV: We sample the shocks together to ensure that the [1] and [0] iteration have the same sshocks

            if f in impulse_times:
              # 4: Replace the shock for the kkth variable to be 0 or 1, for the observations in impulse_times vector
              sim_shock[kk] = shock_level
            
            ### 5: Convert shock back into residual, add this to the series
            sim_resid = np.matmul(sim_shock, c_t.T)
            fcast[f, :, kk, shock_level] = pred + sim_resid

            f = f+1  

          else: 
            print(f'Exploded at simulation time step {f}')
            break
        
    return fcast, fcast_cov_mat, sim_shocks

  # @title Old IRF Simulation Wrapper Function
  # Returns: fcast, None, sim_shocks
  def simulate_irf_paths_old(self, Y_train, Y_test, results, device):

    n_var = self.n_var

    # Get the residuals - resiudals are the average of the different bootstrap iterations
    oos_res = Y_test - results['pred']
    oob_res_save = Y_train - results['pred_in']
    oob_res = oob_res_save[~np.isnan(oob_res_save).any(axis=1)]
    oob_res = np.vstack([oob_res, oos_res])

    # Time periods where we create the impulses
    impulse_times = list(range(self.start_shock_time, self.n_simulations, self.endh))
    impulse_times_all = []

    # Shock the system in plausible way (shock few days in a row)
    for e in impulse_times:
      for sj in [1]:
        impulse_times_all.append(e + sj)

    
    # Get covariance matrix of OOB residuals
    cov_mat = np.cov(oob_res.T)
    # Cholesky decomposition of the cov mat into orthogonal shocks
    C = np.linalg.cholesky(cov_mat)
    # Generate orthogonal shocks for each variable

    oob_shocks = np.matmul(oob_res, np.linalg.inv(C.T))

    # Sample from the OOB shocks self.n_simulations times
    simul_shocks = np.zeros((self.n_simulations, n_var))
    for k in range(n_var):
      simul_ids = random.choices(list(range(oob_shocks.shape[0])), k = self.n_simulations)
      simul_shocks[:, k] = oob_shocks[simul_ids, k]

    # Simul_shocks_old is the original 
    simul_shocks_old = simul_shocks.copy()

    fcast = np.zeros((self.n_simulations, n_var, n_var, 3))
    fcast[:] = np.nan

    for kk in range(n_var):
      for shock_level in [0, 1]:
        simul_shocks = simul_shocks_old.copy()

        # Replace the shock for the kkth variable to be 0 or 1, for the observations in impulse_times vector
        simul_shocks[impulse_times, kk] = shock_level

        #print(f'simul shocks before cholesky, var {kk}', np.cov(simul_shocks.T))
        # Bring the shocks through the C matrix to get residuals

        simul_resids = np.matmul(simul_shocks, C.T)

        #print(f'simul residuals after cholesky, var {kk}', np.cov(simul_shocks.T))
        # Simulate forward, using past values
        new_in_linear = np.zeros(self.n_lag_linear * n_var)
        new_in_nonlinear = np.zeros(self.n_lag_d * n_var)

        # Initialize new data at 0
        fcast[0, :, kk, shock_level] = np.zeros((n_var))

        for f in range(1, self.num_simulations):
          if f % 100 == 0:
            print(f, datetime.now())

          # Add the newly observed data to the model (new variables become the L0, appended to front,
          # while we drop the most lagged variables)
          new_in_linear = np.hstack([fcast[f-1, :, kk, shock_level], new_in_linear[:(len(new_in_linear) - n_var)]])
          new_in_nonlinear = np.hstack([fcast[f-1, :, kk, shock_level], new_in_nonlinear[:(len(new_in_nonlinear) - n_var)]])
          
          # Generate MARX transformed variables - for that one new day
          new_data_marx = new_in_nonlinear.copy()

          for lag in range(2, self.n_lag_d + 1):
            for var in range(n_var):
              who_to_avg = list(range(var, n_var * (lag - 1) + var + 1, n_var))
              new_data_marx[who_to_avg[-1]] = new_in_nonlinear[who_to_avg].mean()

          # Combine the first n_lag_linear lags, with the MARX data, to get the full 325-dim input vector
          new_data_all = np.hstack([new_in_linear, new_data_marx])
          new_data_all = np.expand_dims(new_data_all, axis = 0)

          if np.any(np.isnan(new_data_all)) == False and np.all(np.isfinite(new_data_all)) == True:
            # Get the prediction (NEW USING predict_nn function)
            pred = self.predict_nn_old(results, new_data_all, device)
            print('pred', pred)
          else: 
            print(f'Exploded at simulation time step {f}')
            break

          # Append the new value to fcasts
          fcast[f, :, kk, shock_level] = pred + simul_resids[f, :]

    return fcast, None, simul_shocks_old

  # Compute conditional IRFs for ONE repeat/ONE experiment - takes in VARNN estimation results
  def get_irfs_wrapper(self, Y_train, Y_test, results):
    if self.forecast_method == 'old':
      fcast, fcast_cov_mat, sim_shocks = self.simulate_irf_paths_old(Y_train, Y_test, results, self.device)
    else:
      fcast, fcast_cov_mat, sim_shocks = self.simulate_irf_paths_new(Y_train, Y_test, results, self.device)

    return fcast, fcast_cov_mat, sim_shocks



  def load_results(self, file_name):

    folder_path = f'../results/{self.experiment_name}'
    file_path = f'{folder_path}/{file_name}.npz'

    out = np.load(file_path, allow_pickle = True)
    FCAST = out['fcast']
    FCAST_COV_MAT = out['fcast_cov_mat']
    # If there is only one outer bootstrap to load
    FCAST = np.expand_dims(FCAST, axis = 0)

    self.FCAST = FCAST
    self.FCAST_COV_MAT = FCAST_COV_MAT


  # @DEV: Meant to be called from outside passing in saved IRF results from multiple outer bootstraps
  # Even though the IRFUnconditional class is meant to generate only one outer bootstrap. See in the future if we need to refactor to reconcile this inconsistency
  def evaluate_unconditional_irfs(self):

    FCAST = self.FCAST
    FCAST_COV_MAT = self.FCAST_COV_MAT
    num_outer_bootstraps = FCAST.shape[0]

    n_var = self.n_var
    # Remove exploding values
    explode_indices = np.argwhere(np.abs(FCAST) > 5)
    start_index_to_remove_list = pd.DataFrame(explode_indices).groupby(0)[1].min()
    for i, start_index_to_remove in start_index_to_remove_list.iteritems():
      print('Exploding: ', i, start_index_to_remove)
      FCAST[i, start_index_to_remove:, :, :, :] = np.nan

    # Calculate the difference in the variable values with shock and without shock

    # Fcast: (bootstrap num x num inner shocks x timestep x response var x shock var x shock level)
    DIFF = np.zeros((num_outer_bootstraps, len(self.randoms) -1, self.endh, n_var, n_var))
    DIFF[:] = np.nan

    for kk in range(self.n_var): # Shock variable
      for k in range(self.n_var): # Response variable
        for h in range(self.endh):
          # Get the indices of same horizon to average from
          those = [e + h for e in self.randoms if e + h < self.num_simulations]
          #if len(those) == len(randoms):
            # difference in the forecasts
          DIFF[:, :, h, k, kk] = FCAST[:, those[:(len(self.randoms) -1)], k, kk, 1] - FCAST[:, those[:(len(self.randoms) -1)], k, kk, 0]

    # Get the cumulative differences
    CUM_DIFF = np.cumsum(DIFF, axis = 2)

    # Get the IRFs and cumulative IRFs

    irf_mat = np.zeros((self.n_var, self.n_var, self.endh, 3))

    for kk in range(self.n_var): # Shock variable
      for k in range(self.n_var): # Response variable
        for h in range(self.endh):
          # Get the indices of same horizon to average from
          # difference in the forecasts
          irf_mat[kk, k, h, 0] = np.nanquantile(DIFF[:, :, h, k, kk], q = 0.16)
          irf_mat[kk, k, h, 1] = np.nanquantile(DIFF[:, :, h, k, kk], q = 0.5)
          irf_mat[kk, k, h, 2] = np.nanquantile(DIFF[:, :, h, k, kk], q = 0.84)

    irf_cum_mat = np.zeros((self.n_var, self.n_var, self.endh, 3))

    for kk in range(self.n_var): # Shock variable
      for k in range(self.n_var): # Response variable
        for h in range(self.endh):
          # Get the indices of same horizon to average from
          # difference in the forecasts
          irf_cum_mat[kk, k, h, 0] = np.nanquantile(CUM_DIFF[:, :, h, k, kk], q = 0.16)
          irf_cum_mat[kk, k, h, 1] = np.nanquantile(CUM_DIFF[:, :, h, k, kk], q = 0.5)
          irf_cum_mat[kk, k, h, 2] = np.nanquantile(CUM_DIFF[:, :, h, k, kk], q = 0.84)

    self.irf_mat = irf_mat
    self.irf_cum_mat = irf_cum_mat
    self.DIFF = DIFF
    self.CUM_DIFF = CUM_DIFF


  def plot_irfs(self, image_folder_path):

    num_outer_bootstraps = self.FCAST.shape[0]
    fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (4 * self.n_var, 4 * self.n_var), constrained_layout = True)

    for shock_var in range(self.n_var):
      for response_var in range(self.n_var):
        # Plot median
        axs[response_var, shock_var].plot(self.irf_mat[shock_var, response_var, :self.max_h, 1], lw = 1.5, color = 'black')

        #Plot all bootstraps 
        if self.plot_all_bootstraps == True:
          for b in range(num_outer_bootstraps):
            for r in range(len(self.randoms) - 1):
              axs[response_var, shock_var].plot(self.DIFF[b, r, :self.max_h, response_var, shock_var], lw = 0.5, alpha = 0.2)

        # Plot confidence bands
        #axs[response_var, shock_var].fill_between(range(h), irf_mat[shock_var, response_var, :h, 0], irf_mat[shock_var, response_var, :h, 2], alpha = 0.5)

        # Plot confidence bands
        axs[response_var, shock_var].plot(self.irf_mat[shock_var, response_var, :self.max_h, 0], lw = 1.5, color = 'black', ls = '--')
        axs[response_var, shock_var].plot(self.irf_mat[shock_var, response_var, :self.max_h, 2], lw = 1.5, color = 'black', ls = '--')

        axs[response_var, shock_var].set_title(f'Shock: {self.var_names[shock_var]}, Response: {self.var_names[response_var]}')
        axs[response_var, shock_var].set_xlabel('Periods')
        axs[response_var, shock_var].set_ylabel('Impulse Response')
        #axs[response_var, shock_var].set_ylim((-0.5, 1))

    image_file = f'{image_folder_path}/irf.png'
    plt.savefig(image_file)

    print(f'IRFs plotted at {image_file}')

  def plot_cumulative_irfs(self, image_folder_path):

    num_outer_bootstraps = self.FCAST.shape[0]
    fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (4 * self.n_var, 4 * self.n_var), constrained_layout = True)

    for shock_var in range(self.n_var):
      for response_var in range(self.n_var):
        # Plot median
        axs[response_var, shock_var].plot(self.irf_cum_mat[shock_var, response_var, :self.max_h, 1], lw = 1.5, color = 'black')

        #Plot all bootstraps 
        if self.plot_all_bootstraps == True:
          for b in range(num_outer_bootstraps):
            for r in range(len(self.randoms) - 1):
              axs[response_var, shock_var].plot(self.CUM_DIFF[b, r, :self.max_h, response_var, shock_var], lw = 0.5, alpha = 0.2)

        # Plot confidence bands
        #axs[response_var, shock_var].fill_between(range(h), irf_mat[shock_var, response_var, :h, 0], irf_mat[shock_var, response_var, :h, 2], alpha = 0.5)

        # Plot confidence bands
        axs[response_var, shock_var].plot(self.irf_cum_mat[shock_var, response_var, :self.max_h, 0], lw = 1.5, color = 'black', ls = '--')
        axs[response_var, shock_var].plot(self.irf_cum_mat[shock_var, response_var, :self.max_h, 2], lw = 1.5, color = 'black', ls = '--')

        axs[response_var, shock_var].set_title(f'Shock: {self.var_names[shock_var]}, Response: {self.var_names[response_var]}')
        axs[response_var, shock_var].set_xlabel('Periods')
        axs[response_var, shock_var].set_ylabel('Impulse Response')

    image_file = f'{image_folder_path}/cumulative_irf.png'
    plt.savefig(image_file)

    print(f'Cumulative IRFs plotted at {image_file}')