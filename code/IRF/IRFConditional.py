import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class IRFConditional:

  def __init__(self, results, irf_params):
    self.n_betas = irf_params['n_betas']
    self.n_lags = irf_params['n_lags']
    self.n_var = irf_params['n_var']
    self.max_h = irf_params['max_h']
    self.var_names = irf_params['var_names']

    # Sum hemis to get BETAS: (n_obs, n_betas, n_bootstraps, n_var)
    self.BETAS = np.sum(results['betas'], axis = -1)
    self.SIGMAS = results['sigmas']
    self.IRFS = None

    # Computes the IRFs
    self.get_irfs()

  # @title Function to get IRF for One Draw
  def _get_irf_draw(self, beta_draw, cov_mat_draw = None):

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

    # # Get the median of the IRF draws
    # IRFS_median = np.nanmedian(IRFS, axis = 1)
    
    # return IRFS_median


  def plot_irfs(self, image_folder_path, experiment_id):

    # Take the median 
    IRFS_median = np.nanmedian(self.IRFS, axis = 1)

    times_to_draw = [90, 210, 330, 450]
    times_to_draw_labels = [1968, 1978, 1988, 1998]
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(self.n_var, self.n_var, constrained_layout = True, figsize = (4 * self.n_var, 4 * self.n_var))

    for k in range(self.n_var):
      for kk in range(self.n_var):
        for i in range(len(times_to_draw)):
          irf_df = IRFS_median[times_to_draw[i], k, kk, :]
          ax[kk, k].plot(irf_df, label = times_to_draw_labels[i], color = cmap(i))

          #irf_actual_df = IRFS_actual[times_to_draw[i], k, kk, :]
          #ax[kk, k].plot(irf_actual_df, label = str(times_to_draw_labels[i]) + ' Actual', ls = '--', color = cmap(i))
          ax[kk, k].set_xlabel('Horizon')
          ax[kk, k].set_ylabel('Impulse Response')
          ax[kk, k].axhline(y = 0, color = 'black', ls = '--')
          ax[kk, k].set_title(f'{self.var_names[k]} -> {self.var_names[kk]}')
        if k == 0 and kk == 0:
          ax[kk, k].legend()

    image_path = f'{image_folder_path}/irf_conditional_{experiment_id}.png'
    plt.savefig(image_path)

    print(f'Conditional IRF saved at {image_path}')


