from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random

from EconDL.utils import invert_scaling
from EconDL.predict_nn import predict_nn_new, predict_nn_old


class IRFUnconditional:

  def __init__(self, irf_params, device = None):

    self.n_lag_linear = irf_params['n_lag_linear']
    self.n_lag_d = irf_params['n_lag_d']
    self.n_var = irf_params['n_var']
    self.max_h = irf_params['max_h']
    self.var_names = irf_params['var_names']
    
    self.num_simulations = irf_params['num_simulations']
    self.start_shock_time = irf_params['start_shock_time']
    self.endh = irf_params['endh']
    self.forecast_method = irf_params['forecast_method']
    
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
            
#            print('new_data_all', new_data_all)
            pred, cov = predict_nn_new(results, new_data_all, device)

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

            # print('pred', pred)
            # print('sim_resid', sim_resid)

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
            pred = predict_nn_old(results, new_data_all, device)
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
