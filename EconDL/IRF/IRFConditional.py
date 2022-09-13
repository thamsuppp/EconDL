import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# plotly api key
plotly_api_key = 'Dns1gp04h4QpiskQPFT3'
chart_studio.tools.set_credentials_file(username= 'thamsuppp', api_key = plotly_api_key)

class IRFConditional:

  def __init__(self, results, irf_params):
    self.n_betas = irf_params['n_betas']
    self.n_lags = irf_params['n_lags']
    self.n_var = irf_params['n_var']
    self.max_h = irf_params['max_h']
    self.var_names = irf_params['var_names']
    self.test_exclude_last = irf_params['test_exclude_last']
    self.dataset_name = irf_params['dataset']

    # Sum hemis to get BETAS: (n_obs, n_betas, n_bootstraps, n_var)
    self.BETAS = np.nansum(results['betas'], axis = -1)
    self.SIGMAS = results['sigmas']
    self.IRFS = None

    self.experiment_id = irf_params['experiment_id']
    self.experiment_name = irf_params['experiment_name']
    
    # Computes the IRFs
    self.get_irfs()

  # @title Function to get IRF for One Draw
  def _get_irf_draw(self, beta_draw, cov_mat_draw = None):
    
    #try:
    # Rearranging the betas so they fit into A matrix
    constant = beta_draw[:, 0]

    A_L_mats = []
    for i in range(self.n_lags):
      A_L_mats.append(beta_draw[:, list(range(i+1, self.n_betas, self.n_lags))])
    
    # Form the A matrix
    A = np.array(A_L_mats)

    # Generalized Code to construct companion matrix
    p, k = A.shape[0], A.shape[1]
    comp_mat = np.zeros((p*k, p*k))

    for i in range(p):
      comp_mat[0:k, (k*i):(k*(i+1))] = A[i, :, :]

    for i in range((p-1)*k):
      comp_mat[k+i, i] = 1

    # Contemporaneous covariance matrix: k x k
    if cov_mat_draw is None:
      cov_mat_draw = np.identity(self.n_var)

    # Do Cholesky decomposition
    if np.isnan(cov_mat_draw[0, 0]) == True:
      irf_draw = np.zeros((self.n_var, self.n_var, self.max_h))
      irf_draw[:] = np.nan
      return irf_draw
    else:
      C = np.linalg.cholesky(cov_mat_draw)

    # Construct the companion C matrix
    C_mat = np.zeros((p*k, p*k))
    for i in range(p):
      C_mat[(k*i):(k*(i+1)), (k*i):(k*(i+1))] = C

    # shock var x response var x h
    irf_draw = np.zeros((self.n_var, self.n_var * self.n_lags, self.max_h))

    for h in range(self.max_h):
      for var in range(self.n_var):
        shock_vec = np.zeros((self.n_var * self.n_lags,))
        shock_vec[var] = 1.0
        irf_draw[var, :, h] = np.matmul(np.linalg.matrix_power(comp_mat, h), np.matmul(C_mat, shock_vec))

    # Cut off the repeated second half of the IRF matrix
    irf_draw = irf_draw[:, 0:(self.n_var), :]

    return irf_draw
    # except:
    #   # Linalg error
    #   print('cov_mat_draw non PSD', cov_mat_draw)
    #   return np.zeros((self.n_var, self.n_var, self.max_h))

  # @title Computes the conditional IRFs for all bootstrap draws, updates self.IRFS
  # Note: If you want to use estimated covariance matrix, then change cov_mat_draw parameter to cov_mat_draw, otherwise None makes it identity matrix
  def get_irfs(self):

    BETAS = self.BETAS
    SIGMAS = self.SIGMAS

    n_obs = BETAS.shape[0]
    n_bootstraps = BETAS.shape[2]
    n_var = BETAS.shape[3]

    IRFS = np.zeros((n_obs, n_bootstraps, n_var, n_var, self.max_h))
    IRFS[:] = np.nan


    for t in range(n_obs):
      if t % 100 == 0:
        print(f'Simulation timestep {t} at {datetime.now()}')
      for boot in range(n_bootstraps):
          
        # Can change BETAS_IN or BETAS
        beta_draw = BETAS[t, :, boot, :].T

        # # Get the time-invariant cov mat
        #cov_mat_draw = COV_MATS[boot, :, :]
        # Get the time-varying cov mat
        cov_mat_draw = SIGMAS[t, :, :, boot]
        irf_draw = self._get_irf_draw(beta_draw, cov_mat_draw = cov_mat_draw)

        IRFS[t, boot, :, :, :] = irf_draw

    self.IRFS = IRFS

    # Save the IRFs
    np.savez(f'IRF_conditional.npz', IRFS = IRFS)

  def plot_irfs_3d(self, image_folder_path):
    # Take the median 
    IRFS_median = np.nanmedian(self.IRFS, axis = 1)
    n_var = IRFS_median.shape[1]
        
    fig = make_subplots(rows = n_var, cols = n_var,
                        subplot_titles = [f'IRF {self.var_names[shock_var]} -> {self.var_names[response_var]}' for response_var in range(n_var) 
                          for shock_var in range(n_var)],
                        specs = [[{'is_3d': True} for e in range(n_var)] for e in range(n_var)],
                        shared_xaxes = False,
                        shared_yaxes = False,
                        horizontal_spacing = 0,
                        vertical_spacing = 0.05
    )

    for shock_var in range(n_var):
      for response_var in range(n_var):
        fig.add_trace(go.Surface(name = self.experiment_name, z = IRFS_median[:, shock_var, response_var, :], 
              showscale = False, showlegend = True, 
              colorscale = 'RdBu', cmid = 0, opacity = 0.75),
              row = response_var + 1, col = shock_var + 1)

    fig.update_scenes(xaxis_title = 'Horizon',
                      yaxis_title = 'Time', 
                      zaxis_title = 'Value',
                      camera = {
                      'up': {'x': 0, 'y': 0, 'z': 1},
                      'center': {'x': 0, 'y': 0, 'z': 0},
                      'eye': {'x': 1.25, 'y': -1.5, 'z': 0.75}
                      })


    fig.update_layout(title = f'Experiment {self.experiment_id} ({self.experiment_name}) Conditional IRF', autosize=False,
                      width = 350 * n_var, height = 350 * n_var,
                      margin=dict(l=25, r=25, b=65, t=90))

    image_path = f'{image_folder_path}/irf_conditional_3d_{self.experiment_id}.html'
    fig.write_html(image_path)


  def plot_irfs(self, image_folder_path):

    # Take the median 
    IRFS_median = np.nanmedian(self.IRFS, axis = 1)

    if self.dataset_name == 'monthly':
      times_to_draw = [90, 210, 330, 450]
      times_to_draw_labels = [1968, 1978, 1988, 1998]
    elif self.dataset_name == 'quarterly':
      times_to_draw = [30, 70, 110, 150]
      times_to_draw_labels = [1968, 1978, 1988, 1998]
    else:
      times_to_draw = list(np.linspace(0, self.IRFS.shape[0], 6, dtype = int)[1:-1])
      times_to_draw_labels = times_to_draw

    cmap = plt.cm.tab10

    # Plot IRF
    fig, ax = plt.subplots(self.n_var, self.n_var, constrained_layout = True, figsize = (4 * self.n_var, 4 * self.n_var))

    for shock_var in range(self.n_var):
      for response_var in range(self.n_var):
        for i in range(len(times_to_draw)):
          irf_df = IRFS_median[times_to_draw[i], shock_var, response_var, :]
          ax[response_var, shock_var].plot(irf_df, label = times_to_draw_labels[i], color = cmap(i))

          #irf_actual_df = IRFS_actual[times_to_draw[i], k, kk, :]
          #ax[kk, k].plot(irf_actual_df, label = str(times_to_draw_labels[i]) + ' Actual', ls = '--', color = cmap(i))
          ax[response_var, shock_var].set_xlabel('Horizon')
          ax[response_var, shock_var].set_ylabel('Impulse Response')
          ax[response_var, shock_var].axhline(y = 0, color = 'black', ls = '--')
          ax[response_var, shock_var].set_title(f'{self.var_names[shock_var]} -> {self.var_names[response_var]}')
        if response_var == 0 and shock_var == 0:
          ax[response_var, shock_var].legend()

    fig.suptitle(f'Experiment {self.experiment_id} ({self.experiment_name}) Conditional IRF', fontsize = 16)
    image_path = f'{image_folder_path}/irf_conditional_{self.experiment_id}.png'
    plt.savefig(image_path)

    # Plot cumulative IRF
    fig, ax = plt.subplots(self.n_var, self.n_var, constrained_layout = True, figsize = (4 * self.n_var, 4 * self.n_var))

    for shock_var in range(self.n_var):
      for response_var in range(self.n_var):
        for i in range(len(times_to_draw)):
          cum_irf_df = np.cumsum(IRFS_median[times_to_draw[i], shock_var, response_var, :], axis = -1)
          ax[response_var, shock_var].plot(cum_irf_df, label = times_to_draw_labels[i], color = cmap(i))

          ax[response_var, shock_var].set_xlabel('Horizon')
          ax[response_var, shock_var].set_ylabel('Impulse Response')
          ax[response_var, shock_var].axhline(y = 0, color = 'black', ls = '--')
          ax[response_var, shock_var].set_title(f'{self.var_names[shock_var]} -> {self.var_names[response_var]}')
        if response_var == 0 and shock_var == 0:
          ax[response_var, shock_var].legend()

    fig.suptitle(f'Experiment {self.experiment_id} ({self.experiment_name}) Cumulative Conditional IRF', fontsize = 16)
    image_path = f'{image_folder_path}/irf_cum_conditional_{self.experiment_id}.png'
    plt.savefig(image_path)

    print(f'Conditional IRF plotted at {image_path}')

  # Compare the IRFs over all time periods for each horizon
  # normalize: normalize by the 0th shock being 1
  def plot_irfs_over_time(self, image_folder_path, normalize = True):

    # Take the median 
    IRFS_median = np.nanmedian(self.IRFS, axis = 1)
    # Exclude last samples if relevant
    if self.test_exclude_last != 0:
      IRFS_median = IRFS_median[:-self.test_exclude_last, :, :, :]

    cmap = plt.cm.Reds(np.linspace(1,0,6))
    fig, ax = plt.subplots(self.n_var, self.n_var, constrained_layout = True, figsize = (6 * self.n_var, 4 * self.n_var))

    for shock_var in range(self.n_var):
      for response_var in range(self.n_var):
        for h in [0,1,2,3,4]:
          irf_df = IRFS_median[:, shock_var, response_var, h]
          if normalize == True:
            irf_df = irf_df / IRFS_median[:, shock_var, response_var, 0] # Divide IRF by the time-0 of 
          ax[response_var, shock_var].plot(irf_df, label = f'h={h}', color = cmap[h], lw = 1)

          #irf_actual_df = IRFS_actual[:, k, kk, h]
          #ax[kk, k].plot(irf_actual_df, label = f'h={h} Actual', color = 'black')
          ax[response_var, shock_var].set_xlabel('Horizon')
          ax[response_var, shock_var].set_ylabel('Impulse Response')
          ax[response_var, shock_var].axhline(y = 0, color = 'black', ls = '--')
          ax[response_var, shock_var].set_title(f'{self.var_names[shock_var]} -> {self.var_names[response_var]}')
        if response_var == 0 and shock_var == 0:
          ax[response_var, shock_var].legend()

    fig.suptitle(f'Experiment {self.experiment_id} ({self.experiment_name}) Conditional IRF Over Time', fontsize = 16)
    image_file = f'{image_folder_path}/irf_conditional_over_time_{self.experiment_id}.png'
    plt.savefig(image_file)
    print(f'Conditional IRF over time plotted at {image_file}')


