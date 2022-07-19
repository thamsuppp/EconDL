
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

### Experiment-level class to evaluate unconditional IRFs
class IRFUnconditionalEvaluation:

  def __init__(self, FCAST, FCAST_COV_MAT, irf_params):
    self.FCAST = FCAST
    self.FCAST_COV_MAT = FCAST_COV_MAT

    self.n_lag_linear = irf_params['n_lag_linear']
    self.n_lag_d = irf_params['n_lag_d']
    self.n_var = irf_params['n_var']
    self.max_h = irf_params['max_h']
    self.var_names = irf_params['var_names']
    
    self.num_simulations = irf_params['num_simulations']
    self.start_shock_time = irf_params['start_shock_time']
    self.endh = irf_params['endh']
    self.randoms = list(range(self.start_shock_time, self.num_simulations, self.endh))
    self.plot_all_bootstraps = irf_params['plot_all_bootstraps']
  
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


  def plot_irfs(self, image_folder_path, experiment_id):

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

    image_file = f'{image_folder_path}/irf_{experiment_id}.png'
    plt.savefig(image_file)

    print(f'Unconditional IRFs plotted at {image_file}')

  def plot_cumulative_irfs(self, image_folder_path, experiment_id):

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

    image_file = f'{image_folder_path}/cumulative_irf_{experiment_id}.png'
    plt.savefig(image_file)

    print(f'Cumulative Unconditional IRFs plotted at {image_file}')