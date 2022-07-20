import numpy as np
import torch 
from EconDL.utils import invert_scaling

# @title Predict NN Function (Joint Estimation)
# Returns pred, cov
def predict_nn_new(results, newx, bootstraps_to_ignore = [], device = None):

  scale_output = results['scale_output']
  if results['prior_shift'] == True:
    # Make the predictions for the prior shift - for the new data
    ps_model = results['ps_model']
    x_pos_ps = results['x_pos_ps']

    newx_temp = np.hstack([np.ones((newx.shape[0], 1)), newx[:, x_pos_ps]])
    pred_oos_adj = ps_model.predict(newx_temp)

    pred_oos_adj = pred_oos_adj * results['prior_shift_strength']
  
  else:
    pred_oos_adj = 0.0

  # Scale the new x
  if results['standardize'] == True:
    scaler_x = scale_output['scaler_x']
    newx = scaler_x.transform(newx)

  newx_tensor = torch.tensor(newx, dtype = torch.float).to(device)

  num_inner_bootstraps = len(results['trained_model'])
  n_var = results['pred_in'].shape[1]

  # Prediction matrix: n_observations x num_inner_bootstraps x n_vars
  pred_mat = np.zeros((newx.shape[0], num_inner_bootstraps, n_var))
  pred_mat[:] = np.nan

  # Cov mat: n_observations x num_inner_bootstraps x n_vars x n_vars
  cov_mat = np.zeros((newx.shape[0], num_inner_bootstraps, n_var, n_var))
  cov_mat[:] = np.nan

  for i in range(num_inner_bootstraps):
    # Use new feature matrix to get predictions for next period
    model_for_prediction = results['trained_model'][i]
    # Assuming newx is 2D (n_obs x n_x_vars)
    pred, precision, _, _, _ = model_for_prediction(newx_tensor)
    pred = pred.detach().cpu().numpy()

    if results['standardize'] == True:
      pred = invert_scaling(pred, scale_output['mu_y'], scale_output['sigma_y'])
    pred_mat[:, i, :] = pred

    precision = precision.detach().cpu().numpy()
    sigma = np.linalg.inv(precision) # Standardized cov mat
    
    if results['standardize'] == True:
      for j in range(n_var): # Unscale cov mat back to original scaling
        sigma[:, j, :] = sigma[:, j, :] * scale_output['sigma_y'][j]
        sigma[:, :, j] = sigma[:, :, j] * scale_output['sigma_y'][j]
    
    if sigma[0, 0 , 0] > 0:
      cov_mat[:, i, :, :] = sigma
    else:
      print(f'Non-PSD cov mat found at bootstrap {i}')
      if i not in bootstraps_to_ignore:
        bootstraps_to_ignore.append(i)

    #print(f'Bootstrap {i} Cov Mat: ', sigma)

  # Take mean BEFORE unscaling (REVISIT IF WE NEED TO FLIP ORDER)
  pred = np.nanmedian(pred_mat, axis = 1)
  # Add back the oos adj

  pred = pred + pred_oos_adj
  cov = np.nanmedian(cov_mat, axis = 1)

  return pred, cov, bootstraps_to_ignore

# @title Predict NN Function OLD (Non-Joint Estimation)
# Returns: pred
def predict_nn_old(self, results, newx, device):

  scale_output = results['scale_output']
  if results['prior_shift'] == True:
    # Make the predictions for the prior shift - for the new data
    ps_model = results['ps_model']
    x_pos_ps = results['x_pos_ps']

    newx_temp = np.hstack([np.ones((newx.shape[0], 1)), newx[:, x_pos_ps]])
    pred_oos_adj = ps_model.predict(newx_temp)

    pred_oos_adj = pred_oos_adj * results['prior_shift_strength']
  
  else:
    pred_oos_adj = 0.0

  # Scale the new x
  if results['standardize'] == True:
    scaler_x = scale_output['scaler_x']
    newx = scaler_x.transform(newx)

  newx_tensor = torch.tensor(newx, dtype = torch.float).to(device)

  num_inner_bootstraps = len(results['trained_model'])
  # Prediction matrix: n_observations x num_inner_bootstraps x n_vars
  pred_mat = np.zeros((newx.shape[0], num_inner_bootstraps, results['pred_in'].shape[1]))

  # Use the trained model in each bootstrap to generate predictions for the next period
  for i in range(num_inner_bootstraps):
    model_for_prediction = results['trained_model'][i]
    # Assuming newx is 2D (n_obs x n_x_vars)
    pred, _, _, _, _ = model_for_prediction(newx_tensor)
    pred = pred.detach().cpu().numpy()
    pred_mat[:, i, :] = pred

  # Take mean BEFORE unscaling (REVISIT IF WE NEED TO FLIP ORDER)
  pred = np.nanmedian(pred_mat, axis = 1)

  # Invert scaling of the prediction
  if results['standardize'] == True:
    pred = invert_scaling(pred, scale_output['mu_y'], scale_output['sigma_y'])
  
  # Add back the oos adj
  pred = pred + pred_oos_adj

  return pred

def predict_ml_model(results, newx):
  scale_output = results['scale_output']

  # Scale the new x
  if results['standardize'] == True:
    scaler_x = scale_output['scaler_x']
    newx = scaler_x.transform(newx)

  # Prediction matrix: n_observations x num_inner_bootstraps x n_vars
  pred = np.zeros((newx.shape[0], n_var))
  pred[:] = np.nan

  # For every variable, make prediction using trained model for that variable
  for var in range(n_var):
    model_var = results['trained_model'][var]
    pred[:, var] = model_var.predict(newx)

  # Unstandardize the predictions
  if results['standardize'] == True:
    pred = invert_scaling(pred, scale_output['mu_y'], scale_output['sigma_y'])

  return pred