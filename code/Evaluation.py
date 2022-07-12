import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class Evaluation:
  def __init__(self, run_name, evaluation_params):

    self.run_name = run_name
    self.folder_path = f'../results/{run_name}'

    # Create image folder if not exist yet
    self.image_folder_path = f'{self.folder_path}/images'
    if os.path.isdir(self.image_folder_path) == False:
      os.mkdir(self.image_folder_path)


    self.need_to_combine = evaluation_params['need_to_combine']
    self.is_simulation = evaluation_params['is_simulation']
    self.is_test = evaluation_params['is_test']
    self.multiple_datasets = evaluation_params['multiple_datasets']
    self.plot_all_bootstraps = evaluation_params['plot_all_bootstraps']
    self.sim_dataset = evaluation_params['sim_dataset']
    self.n_var = len(evaluation_params['var_names'])
    self.var_names = evaluation_params['var_names']
    self.beta_names = ['Constant'] + self.var_names
    self.test_size = evaluation_params['test_size']
    self.exps_to_plot = evaluation_params['exps_to_plot']

    # Store the names of every hyperparameter list
    self.experiment_names = []

    self.params = []
    self.BETAS_IN_ALL = None
    self.BETAS_ALL = None
    self.SIGMAS_IN_ALL = None
    self.SIGMAS_ALL = None
    self.PRECISION_IN_ALL = None
    self.PRECISION_ALL = None
    self.CHOLESKY_IN_ALL = None
    self.CHOLESKY_ALL = None
    self.SIGMAS_CONS_ALL = None

    # Load the results and params
    self.load_results(evaluation_params['experiments_to_load'])

  def check_results_sizes(self):
    return {
      'BETAS_IN_ALL': self.BETAS_IN_ALL.shape,
      'BETAS_ALL': self.BETAS_ALL.shape,
      'SIGMAS_IN_ALL': self.SIGMAS_IN_ALL.shape,
      'SIGMAS_ALL': self.SIGMAS_ALL.shape,
      'PRECISION_IN_ALL': self.PRECISION_IN_ALL.shape,
      'PRECISION_ALL': self.PRECISION_ALL.shape,
      'CHOLESKY_IN_ALL': self.CHOLESKY_IN_ALL.shape,
      'CHOLESKY_ALL': self.CHOLESKY_ALL.shape
    }

  def load_results(self, experiments_to_load):

    M_total = len(experiments_to_load)

    for i in range(M_total):
      experiment = experiments_to_load[i]

      compiled_text = 'compiled' if self.need_to_combine == True else 'repeat_0'
      dataset_text = f'_dataset_{self.sim_dataset}' if self.multiple_datasets == True else ''
      load_file = f'{self.folder_path}/params_{experiment}{dataset_text}_{compiled_text}.npz'

      out = np.load(load_file, allow_pickle = True)
      params = out['params'].item()
      self.params.append(params)

      n_lag_linear = params['n_lag_linear']
      num_bootstraps = params['num_bootstrap']
      
      self.experiment_names.append(out['params'].item()['name'])
      BETAS = out['betas']
      BETAS_IN = out['betas_in']
      SIGMAS = out['sigmas']
      SIGMAS_IN = out['sigmas_in']
      PRECISION = out['precision']
      PRECISION_IN = out['precision_in']
      CHOLESKY = out['cholesky']
      CHOLESKY_IN = out['cholesky_in']

      # Estimate time-invariant cov mat from the residuals
      Y_train = out['y']
      PREDS = out['train_preds']
      resids = np.repeat(np.expand_dims(Y_train, axis = 1), PREDS.shape[1], axis = 1) - PREDS

      # For experiments with more than 1 lag, get the ids of the 1st beta to plot
      beta_ids_to_keep = [0] + list(range(1, BETAS_IN.shape[1], n_lag_linear))
      print(beta_ids_to_keep)
      BETAS_IN = BETAS_IN[:, beta_ids_to_keep, :,:,:]
      BETAS = BETAS[:, beta_ids_to_keep, :,:,:]

      if i == 0:
        BETAS_ALL = np.zeros((M_total, BETAS.shape[0], BETAS.shape[1], num_bootstraps, BETAS.shape[3], BETAS.shape[4]))
        BETAS_ALL[:] = np.nan
        # n_models x n_obs x n_betas x n_bootstraps x n_vars x n_hemispheres
        BETAS_IN_ALL = np.zeros((M_total, BETAS_IN.shape[0], BETAS_IN.shape[1], num_bootstraps, BETAS_IN.shape[3], BETAS_IN.shape[4]))
        BETAS_IN_ALL[:] = np.nan 

        # n_models x n_obs x n_vars x n_vars x n_bootstraps
        SIGMAS_ALL = np.zeros((M_total, SIGMAS.shape[0], SIGMAS.shape[1], SIGMAS.shape[2], num_bootstraps))
        SIGMAS_ALL[:] = np.nan
        PRECISION_ALL = np.zeros_like(SIGMAS_ALL)
        PRECISION_ALL[:] = np.nan
        CHOLESKY_ALL = np.zeros((M_total, SIGMAS.shape[0], SIGMAS.shape[1], SIGMAS.shape[2], 2, num_bootstraps))
        CHOLESKY_ALL[:] = np.nan 

        SIGMAS_IN_ALL = np.zeros((M_total, SIGMAS_IN.shape[0], SIGMAS_IN.shape[1], SIGMAS_IN.shape[2], num_bootstraps))
        SIGMAS_IN_ALL[:] = np.nan 
        PRECISION_IN_ALL = np.zeros_like(SIGMAS_IN_ALL)
        PRECISION_IN_ALL[:] = np.nan
        CHOLESKY_IN_ALL = np.zeros_like(CHOLESKY_ALL)
        CHOLESKY_IN_ALL[:] = np.nan

        SIGMAS_CONS_ALL = np.zeros((M_total, SIGMAS.shape[1], SIGMAS.shape[2], num_bootstraps))
        SIGMAS_CONS_ALL[:] = np.nan

      # If >1 hemis, Demean the time hemisphere and add the mean to the endogenous hemisphere
      # (note: the means are the in-sample means, not the oob ones)
      if BETAS.shape[4] > 1:
        time_hemi_means = np.nanmean(BETAS_IN[:,:,:,:,1], axis = 0)
        time_hemi_means_expand = np.repeat(np.expand_dims(time_hemi_means, axis = 0), BETAS.shape[0], axis = 0)
        # BETAS_IN[:, :, :, :, 0] = BETAS_IN[:, :, :, :, 0] + time_hemi_means_expand
        # BETAS_IN[:, :, :, :, 1] = BETAS_IN[:, :, :, :, 1] - time_hemi_means_expand
        # BETAS[:, :, :, :, 0] = BETAS[:, :, :, :, 0] + time_hemi_means_expand
        # BETAS[:, :, :, :, 1] = BETAS[:, :, :, :, 1] - time_hemi_means_expand

      BETAS_ALL[i,:,:,:BETAS_IN.shape[2],:, :BETAS_IN.shape[4]] = BETAS
      BETAS_IN_ALL[i,:,:,:BETAS_IN.shape[2],:, :BETAS_IN.shape[4]] = BETAS_IN
      SIGMAS_ALL[i, :,:,:,:] = SIGMAS
      SIGMAS_IN_ALL[i, :,:,:,:] = SIGMAS_IN
      PRECISION_ALL[i, :,:,:,:] = PRECISION
      PRECISION_IN_ALL[i, :,:,:,:] = PRECISION_IN
      CHOLESKY_ALL[i, :,:,:,:, :] = CHOLESKY
      CHOLESKY_IN_ALL[i, :,:,:,:, :] = CHOLESKY_IN

      for b in range(num_bootstraps):
        SIGMAS_CONS_ALL[i, :,:,b] = pd.DataFrame(resids[:, b, :]).dropna().cov()

    self.BETAS_ALL = BETAS_ALL
    self.BETAS_IN_ALL = BETAS_IN_ALL
    self.SIGMAS_IN_ALL = SIGMAS_IN_ALL
    self.SIGMAS_ALL = SIGMAS_ALL
    self.PRECISION_IN_ALL = PRECISION_IN_ALL
    self.PRECISION_ALL = PRECISION_ALL
    self.CHOLESKY_IN_ALL = CHOLESKY_IN_ALL
    self.CHOLESKY_ALL = CHOLESKY_ALL
    self.SIGMAS_CONS_ALL = SIGMAS_CONS_ALL


  def load_benchmark_results(self):
    pass
    

  # Helper function to plot betas
  def plot_betas_all(self, BETAS, var_names, beta_names, image_file, q = 0.16, title = '', actual = None):

    n_obs = BETAS.shape[0]
    n_betas = BETAS.shape[1]
    n_bootstraps = BETAS.shape[2]
    n_vars = BETAS.shape[3]
    fig, axs = plt.subplots(n_vars, n_betas, figsize = (6 * n_betas, 4 * n_vars), constrained_layout = True)

    for var in range(n_vars):
      for beta in range(n_betas):

        # Get the quantiles
        betas_lcl = np.nanquantile(BETAS[:, beta, :, var], q = q, axis = 1)
        betas_ucl = np.nanquantile(BETAS[:, beta, :, var], q = 1 - q, axis = 1)
        betas_median = np.nanmean(BETAS[:, beta, :, var], axis = 1)

        # betas_median = pd.Series(betas_median, index = dates)
        # betas_lcl = pd.Series(betas_lcl, index = dates)
        # betas_ucl = pd.Series(betas_ucl, index = dates)

        #Plot all bootstraps' paths
        if self.plot_all_bootstraps == True:
          for i in range(n_bootstraps):
            axs[var, beta].plot(BETAS[:, beta, i, var], lw = 0.5, alpha = 0.15)

        axs[var, beta].plot(betas_median, label = f'{var_names[var]} {beta_names[beta]}', lw = 1.5)
        # Plot the confidence bands
        axs[var, beta].fill_between(list(range(BETAS.shape[0])), betas_lcl, betas_ucl, alpha = 0.5)

        # # Plot actual
        # if actual is not None:
        #   actual_swapped = actual.copy()[:, :, [3,0,1,2]]
        #   axs[var, beta].plot(actual_swapped[:, var, beta], color = 'black')

        # Plot the confidence bands (old method, less preferred by PGC)
        # axs[var, beta].plot(betas_lcl, label = f'{var_names[var]} {beta_names[beta]}', lw = 1.5, color = 'black', ls = '--')
        # axs[var, beta].plot(betas_ucl, label = f'{var_names[var]} {beta_names[beta]}', lw = 1.5, color = 'black', ls = '--')

        # Vertical line for train/test split
        #axs[var, beta].axvline(results['oos_index'][0], color = 'black', linewidth = 1, linestyle = '--')
        # Horizontal line for OLS estimation
        #axs[var, beta].axhline(coefs_matrix[var, beta], color = 'green', linewidth = 1)

        #axs[var, beta].set_xticks(dates)

        #Set the y-axis limits to be at the min 10% LCL and max 10% UCL
        axs[var, beta].set_ylim(
            np.nanmin(np.nanquantile(BETAS[:, beta, :, var], axis = -1, q = 0.14)),
            np.nanmax(np.nanquantile(BETAS[:, beta, :, var], axis = -1, q = 0.86))
        )

        axs[var, beta].set_title(f'{var_names[var]}, {beta_names[beta]}')
        axs[var, beta].set_xlabel('Time')
        axs[var, beta].set_ylabel('Coefficient')

    fig.suptitle(title, fontsize=16)
    plt.savefig(image_file)

    print(f'Betas plotted at {image_file}')

  def evaluate_TVPs(self):
    # Plot individual hemisphere and summd betas
    if self.is_test == False:
      BETAS_ALL_PLOT = self.BETAS_IN_ALL[:, :-self.test_size,:,:,:]
      #coefs_tv_plot = coefs_tv[:-test_size, :, :] if is_simulation == True else None
    else:
      BETAS_ALL_PLOT = self.BETAS_ALL[:, -self.test_size:,:,:,:]
      #coefs_tv_plot = coefs_tv[-test_size:, :, :] if is_simulation == True else None

    for i in self.exps_to_plot:
      for hemi in range(2):
        image_file = f'{self.image_folder_path}/betas_{i}_hemi_{hemi}.png'
        self.plot_betas_all(BETAS_ALL_PLOT[i, :, :, :, :, hemi], self.var_names, self.beta_names, image_file, q = 0.16, title = f'Experiment {i}, Hemisphere {hemi}', actual = None)
      
      image_file = f'{self.image_folder_path}/betas_{i}_sum.png'
      self.plot_betas_all(np.sum(BETAS_ALL_PLOT[i, :, :, :, :,:], axis = -1), self.var_names, self.beta_names, image_file, q = 0.16, title = f'Experiment {i} Betas, Sum', actual = None)

  def evaluate_precision(self):

      # Don't show test (change this code to show in-sample)
    if self.is_test == False:
      PRECISION_ALL_PLOT = self.PRECISION_ALL[:, :-self.test_size,:,:,:]
    else:
      PRECISION_ALL_PLOT = self.PRECISION_ALL[:, -self.test_size:,:,:,:]

    for i in self.exps_to_plot:
      fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (6 * self.n_var, 4 * self.n_var), constrained_layout = True)

      for row in range(self.n_var):
        for col in range(self.n_var):
          
          # Plot every bootstrap's value
          if self.plot_all_bootstraps == True:
            for b in range(PRECISION_ALL_PLOT.shape[4]):
              axs[row, col].plot(PRECISION_ALL_PLOT[i, :, row, col, b], lw = 0.5, alpha = 0.15, label = i)

          axs[row, col].plot(np.nanmedian(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1), color = 'black')
          axs[row, col].set_title(f'{self.var_names[row]}, {self.var_names[col]}')
          axs[row, col].set_xlabel('Time')
          axs[row, col].set_ylabel('Coefficient')

          sigmas_lcl = np.nanquantile(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.16)
          sigmas_ucl = np.nanquantile(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.84)
          axs[row, col].fill_between(list(range(PRECISION_ALL_PLOT.shape[1])), sigmas_lcl, sigmas_ucl, alpha = 0.8)

          # Set the y-axis limits to be at the min 10% LCL and max 10% UCL
          # axs[row, col].set_ylim(
          #     np.nanmin(np.nanquantile(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.1)),
          #     np.nanmax(np.nanquantile(PRECISION_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.9))
          # )

      fig.suptitle(f'Experiment {i} Precision', fontsize=16)
      image_file = f'{self.image_folder_path}/precision_{i}.png'
      plt.savefig(image_file)

      print(f'Precision plotted at {image_file}')

  def evaluate_cholesky(self):
    # Don't show test (change this code to show in-sample)
    if self.is_test == False:
      CHOLESKY_ALL_PLOT = self.CHOLESKY_ALL[:, :-self.test_size,:,:,:,:]
    else:
      CHOLESKY_ALL_PLOT = self.CHOLESKY_ALL[:, -self.test_size:,:,:,:,:]

    for i in self.exps_to_plot:
      fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (6 * self.n_var, 4 * self.n_var), constrained_layout = True)

      for row in range(self.n_var):
        for col in range(self.n_var):
          for hemi in range(CHOLESKY_ALL_PLOT.shape[-2]):
            
            # Plot every bootstrap's value
            if self.plot_all_bootstraps == True:
              for b in range(CHOLESKY_ALL_PLOT.shape[4]):
                axs[row, col].plot(CHOLESKY_ALL_PLOT[i, :, row, col, b], lw = 0.5, alpha = 0.15, label = i)

            axs[row, col].plot(np.nanmedian(CHOLESKY_ALL_PLOT[i, :, row, col, hemi, :], axis = -1), label = f'Hemi {hemi}')
            axs[row, col].set_title(f'{self.var_names[row]}, {self.var_names[col]}')
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Coefficient')

            sigmas_lcl = np.nanquantile(CHOLESKY_ALL_PLOT[i, :, row, col, hemi, :], axis = -1, q = 0.16)
            sigmas_ucl = np.nanquantile(CHOLESKY_ALL_PLOT[i, :, row, col, hemi, :], axis = -1, q = 0.84)
            axs[row, col].fill_between(list(range(CHOLESKY_ALL_PLOT.shape[1])), sigmas_lcl, sigmas_ucl, alpha = 0.5)

            # Set the y-axis limits to be at the min 10% LCL and max 10% UCL
            # axs[row, col].set_ylim(
            #     np.nanmin(np.nanquantile(CHOLESKY_ALL_PLOT[i, :, row, col, :, 0], axis = -1, q = 0.1)),
            #     np.nanmax(np.nanquantile(CHOLESKY_ALL_PLOT[i, :, row, col, :, 0], axis = -1, q = 0.9))
            # )
            if row == 0 and col == 0:
              axs[row,col].legend()
      fig.suptitle(f'Experiment {i} Cholesky', fontsize=16)
      image_file = f'{self.image_folder_path}/cholesky_{i}.png'
      plt.savefig(image_file)

      print(f'Cholesky plotted at {image_file}')

  def evaluate_sigmas(self):
        
    # Don't show test (change this code to show in-sample)
    if self.is_test == False:
      SIGMAS_ALL_PLOT = self.SIGMAS_ALL[:, :-self.test_size,:,:,:]
      #cov_mat_tv_plot = cov_mat_tv[:-test_size, :, :] if self.is_simulation == True else None

    else:
      SIGMAS_ALL_PLOT = self.SIGMAS_ALL[:, -self.test_size:,:,:,:]
      #cov_mat_tv_plot = cov_mat_tv[-test_size:, :, :] if self.is_simulation == True else None

    for i in self.exps_to_plot:
      fig, axs = plt.subplots(self.n_var, self.n_var, figsize = (6 * self.n_var, 4 * self.n_var), constrained_layout = True)

      for row in range(self.n_var):
        for col in range(self.n_var):
          
          # Plot every bootstrap's value 
          if self.plot_all_bootstraps == True:
            for b in range(self.SIGMAS_IN_ALL.shape[4]):
              axs[row, col].plot(SIGMAS_ALL_PLOT[i, :, row, col, b], lw = 0.5, alpha = 0.15, label = i)

          axs[row, col].plot(np.nanmedian(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1))
          axs[row, col].set_title(f'{self.var_names[row]}, {self.var_names[col]}')
          axs[row, col].set_xlabel('Time')
          axs[row, col].set_ylabel('Coefficient')

          sigmas_lcl = np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.16)
          sigmas_ucl = np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.84)
          axs[row, col].fill_between(list(range(SIGMAS_ALL_PLOT.shape[1])), sigmas_lcl, sigmas_ucl, alpha = 0.5)

          #Plot the actual covariance matrix values
          # if self.is_simulation == True:
          #   axs[row, col].plot(cov_mat_tv_plot[:, row, col], color = 'black')

          # Set the y-axis limits to be at the min 10% LCL and max 10% UCL
          axs[row, col].set_ylim(
              np.nanmin(np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.1)),
              np.nanmax(np.nanquantile(SIGMAS_ALL_PLOT[i, :, row, col, :], axis = -1, q = 0.9))
          )
          # Plot the time-invariant covariance matrix
          axs[row, col].axhline(y = np.nanmedian(self.SIGMAS_CONS_ALL[i, row, col, :]), color = 'red', label = 'Time-Invariant')
      print('Time-Invariant Cov Mat', np.nanmedian(self.SIGMAS_CONS_ALL[i, :,:,:], axis = -1))
      print('Mean Median Time-varying Cov Mat', np.nanmean(np.nanmedian(SIGMAS_ALL_PLOT[i, :, :, :, :], axis = -1), axis = 0))

      fig.suptitle(f'Experiment {i} Sigma', fontsize=16)
      image_file = f'{self.image_folder_path}/sigmas_{i}.png'
      plt.savefig(image_file)

      print(f'Cov Mat plotted at {image_file}')


  def evaluate_one_step_forecasts(results, benchmark_results):
      return {}

  def evaluate_multi_step_forecasts(results, benchmark_results):
      return {}