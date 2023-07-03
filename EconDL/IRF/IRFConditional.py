import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from statsmodels.tsa.api import VAR

import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# plotly api key
plotly_api_key = 'Dns1gp04h4QpiskQPFT3'
chart_studio.tools.set_credentials_file(username= 'thamsuppp', api_key = plotly_api_key)

class IRFConditional:

  def __init__(self, results, irf_params, Y_train, image_folder_path):
    self.n_betas = irf_params['n_betas']
    self.n_lags = irf_params['n_lags']
    self.n_lag_d = irf_params['n_lag_d']
    self.n_var = irf_params['n_var']
    self.max_h = irf_params['max_h']
    self.var_names = irf_params['var_names']
    self.test_size = irf_params['test_size']
    self.test_exclude_last = irf_params['test_exclude_last']
    self.dataset_name = irf_params['dataset']
    self.image_folder_path = image_folder_path

    # Sum hemis to get BETAS: (n_obs, n_betas, n_bootstraps, n_var)
    self.BETAS = np.nansum(results['betas'], axis = -1)
    self.SIGMAS = results['sigmas']
    self.IRFS = None

    self.experiment_id = irf_params['experiment_id']
    self.experiment_name = irf_params['experiment_name']
    self.Y_train = Y_train
    
    # Computes the IRFs
    self.get_irfs()
    
    self.var_irf = None
    self.var_cum_irf = None

    self.get_var_irfs()
    
    
    # Take the median 
    IRFS_median = np.nanmedian(self.IRFS, axis = 1)
    
    # Save the self.var_irf into a file
    np.save(f'{self.image_folder_path}/var_irf_{self.experiment_id}.npy', self.var_irf)
    np.save(f'{self.image_folder_path}/var_cum_irf_{self.experiment_id}.npy', self.var_cum_irf)
    # Save self.IRFs
    np.save(f'{self.image_folder_path}/IRFs_{self.experiment_id}.npy', IRFS_median)
    
    
    # Get a list of dates corresponding to each index
    if self.dataset_name == 'quarterly_new':
      self.dates = pd.date_range(start='1960-06-01', end='2022-07-01', freq='Q')
      if self.test_exclude_last == 0:
        self.dates = self.dates[self.n_lag_d:]
      else:
        self.dates = self.dates[self.n_lag_d:-(self.test_exclude_last)]
    elif self.dataset_name == 'monthly_new':
      self.dates = pd.date_range(start='1960-03-01', end='2022-08-01', freq='M')
      if self.test_exclude_last == 0:
        self.dates = self.dates[self.n_lag_d:]
      else:
        self.dates = self.dates[self.n_lag_d:-(self.test_exclude_last)]
    else:
      self.dates = None
    
    print(f'Dates length: {len(self.dates)}, Start: {self.dates[0]}, End: {self.dates[-1]}')
      
    # Recession dates
    self.recession_dates = [
      ['1960-04-01', '1961-02-01'],
      ['1969-12-01', '1970-11-01'],
      ['1973-11-01', '1975-03-01'],
      ['1980-01-01', '1980-07-01'],
      ['1981-07-01', '1982-11-01'],
      ['1990-07-01', '1991-03-01'],
      ['2001-03-01', '2001-11-01'],
      ['2007-12-01', '2009-06-01'],
      ['2020-02-01', '2020-04-01']
    ]

    
  def get_var_irfs(self):
    var_model = VAR(self.Y_train)
    var_results = var_model.fit(self.n_lags)
    irf = var_results.irf(self.max_h)
    irf_median = irf.orth_irfs
    irf_stderr = irf.stderr(orth = True)

    # (n_horizons, var, var, std errors)
    var_irf = np.zeros((self.max_h + 1, self.n_var, self.n_var, 3))
    var_irf[:] = np.nan
    var_irf[:, :, :, 1] = irf_median
    var_irf[:, :, :, 0] = irf_median - 1 * irf_stderr
    var_irf[:, :, :, 2] = irf_median + 1 * irf_stderr
    

    irf_cum_median = irf.orth_cum_effects
    irf_cum_stderr = irf.cum_effect_stderr(orth = True)
    var_cum_irf = np.zeros((self.max_h + 1, self.n_var, self.n_var, 3))
    var_cum_irf[:] = np.nan
    var_cum_irf[:, :, :, 1] = irf_cum_median
    var_cum_irf[:, :, :, 0] = irf_cum_median - 1 * irf_cum_stderr
    var_cum_irf[:, :, :, 2] = irf_cum_median + 1 * irf_cum_stderr
    
    # Transpose (2,1,0)
    var_irf =  np.transpose(var_irf, (2, 1, 0, 3))
    var_cum_irf = np.transpose(var_cum_irf, (2, 1, 0, 3))
    
    # (var, var, n_horizons, std errors)
    self.var_irf = var_irf
    self.var_cum_irf = var_cum_irf

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

  def plot_irfs_3d(self, image_folder_path, is_test = False):
    
    
    # Take the median 
    if is_test == True:
      IRFS_median = np.nanmedian(self.IRFS[-self.test_size:, :, :, :, :], axis = 1)
    else:
      IRFS_median = np.nanmedian(self.IRFS[:(-self.test_size), :, :, :, :], axis = 1)
      
    n_var = IRFS_median.shape[1]
    
    # Create a plot for each shock variable
    for shock_var in range(n_var):
      
      fig = make_subplots(rows = 1, cols = n_var,
                          subplot_titles = [f'IRF {self.var_names[shock_var]} -> {self.var_names[response_var]}' for response_var in range(n_var)],
                          specs = [[{'is_3d': True} for e in range(n_var)]],
                          shared_xaxes = False,
                          shared_yaxes = False,
                          horizontal_spacing = 0,
                          vertical_spacing = 0.05
      )
      
      for response_var in range(n_var):
        
        irf_this = IRFS_median[:, shock_var, response_var, :]
        initial_shock = IRFS_median[:, response_var, response_var, 0]
        initial_shock = initial_shock.reshape((-1, 1))
        # Divide IRF by time-0 (initial shock)
        irf_this = irf_this / initial_shock
        
        
        # Cumsum to get the cumulative IRF
        irf_this = np.cumsum(irf_this, axis = -1)
        
        fig.add_trace(go.Surface(name = self.experiment_name, z = irf_this,
              showscale = False, showlegend = True, 
              colorscale = 'RdBu', cmid = 0, opacity = 0.75),
              row = 1, col = response_var + 1)
        
      fig.update_scenes(xaxis_title = 'Horizon',
                        yaxis_title = 'Time', 
                        zaxis_title = 'Value',
                        camera = {
                        'up': {'x': 0, 'y': 0, 'z': 1},
                        'center': {'x': 0, 'y': 0, 'z': 0},
                        'eye': {'x': 1.25, 'y': -1.5, 'z': 0.75}
                        })
      
      fig.update_layout(title = f'Experiment {self.experiment_id} ({self.experiment_name}) Cond IRF, Shock on {self.var_names[shock_var]}', autosize=False,
                        width = 350 * n_var, height = 350,
                        margin=dict(l=25, r=25, b=65, t=90))

      image_path = f"{image_folder_path}/irf_conditional_3d_{self.experiment_id}_{self.var_names[shock_var]}{'_test' if is_test == True else ''}.html"
      fig.write_html(image_path)
          
      
        
    # fig = make_subplots(rows = n_var, cols = n_var,
    #                     subplot_titles = [f'IRF {self.var_names[shock_var]} -> {self.var_names[response_var]}' for response_var in range(n_var) 
    #                       for shock_var in range(n_var)],
    #                     specs = [[{'is_3d': True} for e in range(n_var)] for e in range(n_var)],
    #                     shared_xaxes = False,
    #                     shared_yaxes = False,
    #                     horizontal_spacing = 0,
    #                     vertical_spacing = 0.05
    # )

    # for shock_var in range(n_var):
    #   for response_var in range(n_var):
    #     fig.add_trace(go.Surface(name = self.experiment_name, z = IRFS_median[:, shock_var, response_var, :], 
    #           showscale = False, showlegend = True, 
    #           colorscale = 'RdBu', cmid = 0, opacity = 0.75),
    #           row = response_var + 1, col = shock_var + 1)

    # fig.update_scenes(xaxis_title = 'Horizon',
    #                   yaxis_title = 'Time', 
    #                   zaxis_title = 'Value',
    #                   camera = {
    #                   'up': {'x': 0, 'y': 0, 'z': 1},
    #                   'center': {'x': 0, 'y': 0, 'z': 0},
    #                   'eye': {'x': 1.25, 'y': -1.5, 'z': 0.75}
    #                   })


    # fig.update_layout(title = f'Experiment {self.experiment_id} ({self.experiment_name}) Conditional IRF', autosize=False,
    #                   width = 350 * n_var, height = 350 * n_var,
    #                   margin=dict(l=25, r=25, b=65, t=90))

    # image_path = f"{image_folder_path}/irf_conditional_3d_{self.experiment_id}{'_test' if is_test == True else ''}.html"
    # fig.write_html(image_path)


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
        
        # Plot the OLS IRF
        ax[response_var, shock_var].plot(self.var_irf[shock_var, response_var, :, 1], color = 'black', label = 'VAR')
        ax[response_var, shock_var].fill_between(list(range(0, self.max_h)), 
                        self.var_irf[shock_var, response_var, :self.max_h, 0],
                        self.var_irf[shock_var, response_var, :self.max_h, 2],
                        alpha = 0.3, color = 'black')
        
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
        
        # Plot the OLS IRF
        ax[response_var, shock_var].plot(self.var_cum_irf[shock_var, response_var, :, 1], color = 'black', label = 'VAR')
        ax[response_var, shock_var].fill_between(list(range(0, self.max_h)), 
                        self.var_cum_irf[shock_var, response_var, :self.max_h, 0],
                        self.var_cum_irf[shock_var, response_var, :self.max_h, 2],
                        alpha = 0.3, color = 'black')
        
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
      
    # Cumulate the IRFs over horizons to get Cumulative IRFs
    IRFS_median = np.cumsum(IRFS_median, axis = -1)

    cmap = plt.cm.coolwarm(np.linspace(0, 1, 6))
    fig, ax = plt.subplots(self.n_var, self.n_var, constrained_layout = True, figsize = (6 * self.n_var, 4 * self.n_var))

    for shock_var in range(self.n_var):
      for response_var in range(self.n_var):
        
        for h_index, h in enumerate([0, 1, 2, 4, 8, 12]):
          irf_h = IRFS_median[:, shock_var, response_var, h]
          if normalize == True:
            irf_h = irf_h / IRFS_median[:, shock_var, shock_var, 0] # Divide IRF by the time-0 of the shock variable
          ax[response_var, shock_var].plot(self.dates, irf_h, label = f'h={h}', color = cmap[h_index], lw = 1)

          ax[response_var, shock_var].set_xlabel('Horizon')
          ax[response_var, shock_var].set_ylabel('Impulse Response')
          ax[response_var, shock_var].axhline(y = 0, color = 'black', ls = '--')
          ax[response_var, shock_var].set_title(f'{self.var_names[shock_var]} -> {self.var_names[response_var]}')
          
        # Plot the VAR at horizon 12: (var, var, n_horizons, std errors)
        var_irf_h = self.var_cum_irf[shock_var, response_var, 12, 1]
        if normalize == True:
          # Normalize by the VAR's time-0 of the shock variable
          var_irf_h = var_irf_h / self.var_cum_irf[shock_var, shock_var, 0, 1]
                  
        ax[response_var, shock_var].axhline(y = var_irf_h, color = 'black', label = 'VAR h=12', lw = 1)
          
          
        for i in range(len(self.recession_dates)):
          ax[response_var, shock_var].axvspan(self.recession_dates[i][0], self.recession_dates[i][1], alpha=0.2, color='red')
        if response_var == 0 and shock_var == 0:
          ax[response_var, shock_var].legend()

    fig.suptitle(f'Experiment {self.experiment_id} ({self.experiment_name}) Conditional IRF Over Time', fontsize = 16)
    image_file = f'{image_folder_path}/irf_cum_conditional_over_time_{self.experiment_id}.png'
    plt.savefig(image_file)
    print(f'Conditional IRF over time plotted at {image_file}')


