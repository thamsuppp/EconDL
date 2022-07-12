
import matplotlib.pyplot as plt
import numpy as np

def plot_betas_all(BETAS, var_names, beta_names, image_file, q = 0.16, title = '', actual = None):

  print('Test')

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

    #   #Plot all bootstraps' paths
    #   if plot_all_bootstraps == True:
    #     for i in range(n_bootstraps):
    #       axs[var, beta].plot(BETAS[:, beta, i, var], lw = 0.5, alpha = 0.15)

      axs[var, beta].plot(betas_median, label = f'{var_names[var]} {beta_names[beta]}', lw = 1.5)
      # Plot the confidence bands
      # axs[var, beta].fill_between(list(range(BETAS.shape[0])), betas_lcl, betas_ucl, alpha = 0.5)

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

      # #Set the y-axis limits to be at the min 10% LCL and max 10% UCL
      # axs[var, beta].set_ylim(
      #     np.nanmin(np.nanquantile(BETAS[:, beta, :, var], axis = -1, q = 0.14)),
      #     np.nanmax(np.nanquantile(BETAS[:, beta, :, var], axis = -1, q = 0.86))
      # )

      # axs[var, beta].set_title(f'{var_names[var]}, {beta_names[beta]}')
      # axs[var, beta].set_xlabel('Time')
      # axs[var, beta].set_ylabel('Coefficient')

  fig.suptitle(title, fontsize=16)
  plt.savefig(image_file)

def evaluate_TVPs(BETAS_IN_ALL, benchmark_results, evaluation_params, image_folder_path):

  var_names = evaluation_params['var_names']
  test_size = evaluation_params['test_size']
  is_test = evaluation_params['is_test']
  exps_to_plot = evaluation_params['exps_to_plot']

  beta_names = ['Constant'] + var_names

  # Plot individual hemisphere and summd betas
  # if is_test == False:
  #   BETAS_ALL_PLOT = BETAS_IN_ALL[:, :-test_size,:,:,:]
  # else:
  #   BETAS_ALL_PLOT = BETAS_IN_ALL[:, -test_size:,:,:,:]
  BETAS_ALL_PLOT = np.sum(BETAS_IN_ALL[0, :, :, :, :,:], axis = -1)

  image_file = f'{image_folder_path}/betas_sum.png'

  

  plot_betas_all(BETAS_ALL_PLOT, var_names, beta_names, image_file, q = 0.16, actual = None)

  # for i in exps_to_plot:
  #   for hemi in range(2):
  #     image_file = f'{image_folder_path}/betas_{i}_hemi_{hemi}.png'
  #     plot_betas_all(BETAS_ALL_PLOT[i, :, :, :, :, hemi], var_names, beta_names, image_file, q = 0.16, title = '', actual = None)
  #   image_file = f'{image_folder_path}/betas_{i}_sum.png'
  #   plot_betas_all(np.sum(BETAS_ALL_PLOT[i, :, :, :, :,:], axis = -1), var_names, beta_names, image_file, q = 0.16, title = '', actual = None)


def evaluate_one_step_forecasts(results, benchmark_results):
    return {}

def evaluate_multi_step_forecasts(results, benchmark_results):
    return {}