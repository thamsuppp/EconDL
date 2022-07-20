import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
from EconDL.utils import scale_data, invert_scaling

def train_ml_model(X_train, X_test, Y_train, Y_test, nn_hyps, model = 'RF'):

  n_var = Y_train.shape[1]
  if nn_hyps['standardize'] == True:
    print('Standardizing')
    scale_output = scale_data(X_train, Y_train, X_test, Y_test)
    X_train = scale_output['X_train']
    X_test = scale_output['X_test']
    Y_train = scale_output['Y_train']
    Y_test = scale_output['Y_test']

  models = []

  # OOB predictions
  pred = np.zeros_like(Y_train)
  pred[:] = np.nan
  
  for var in range(n_var):
    # Train the model for each variable, and append the trained model
    y_train = Y_train[:, var]

    if model == 'RF':
      model_obj = RandomForestRegressor(random_state = 0, oob_score = True)

      param_list = {
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [25, 50, 100]
      }

      # Tune the model
      rs = RandomizedSearchCV(model_obj, param_list, random_state = 0)
      search = rs.fit(X_train, y_train)

      print(f'Variable {var}, best hyps: {search.best_params_}, time: {datetime.now()}')

      tuned_model_obj = RandomForestRegressor(max_depth = search.best_params_['max_depth'],
                                              n_estimators = search.best_params_['n_estimators'],
                                              oob_score = True)

    elif model == 'XGBoost':
      model_obj = XGBRegressor(subsample = 0.75)
      param_list = {
        'max_depth': [3, 5, 7],
        #'subsample': [0.5, 0.75, 1],
        'n_estimators': [10, 20, 30]
      }

      # Tune the model
      rs = RandomizedSearchCV(model_obj, param_list, random_state = 0)
      search = rs.fit(X_train, y_train)

      print(f'Variable {var}, best hyps: {search.best_params_}, time: {datetime.now()}')

      tuned_model_obj = XGBRegressor(max_depth = search.best_params_['max_depth'],
                                    n_estimators = search.best_params_['n_estimators'],
                                    subsample = 0.75)

    tuned_model_obj.fit(X_train, y_train)
    models.append(tuned_model_obj)

    # Get the predictions (OOB for RF)
    if model == 'RF':
      pred[:, var] = tuned_model_obj.oob_prediction_
    elif model == 'XGBoost':
      pred[:, var] = tuned_model_obj.predict(X_train)

  # Unstandardize preds
  if nn_hyps['standardize'] == True:
    pred = invert_scaling(pred, scale_output['mu_y'], scale_output['sigma_y'])

  return {'trained_model': models,
          'scale_output': scale_output,
          'standardize': nn_hyps['standardize'],
          'pred_in': pred,
          'n_var': n_var}

# def train_ml_model(X_train, X_test, Y_train, Y_test, nn_hyps):

#   n_var = Y_train.shape[1]
#   model = nn_hyps['model']
#   if nn_hyps['standardize'] == True:
#     print('Standardizing')
#     scale_output = scale_data(X_train, Y_train, X_test, Y_test)
#     X_train = scale_output['X_train']
#     X_test = scale_output['X_test']
#     Y_train = scale_output['Y_train']
#     Y_test = scale_output['Y_test']

#   models = []

#   # OOB predictions
#   pred = np.zeros_like(Y_train)
#   pred[:] = np.nan
  
#   for var in range(n_var):
#     # Train the model for each variable, and append the trained model
#     y_train = Y_train[:, var]

#     if model == 'RF':
#       model_obj = RandomForestRegressor(max_depth = 5, random_state = 0, oob_score = True)
#     elif model == 'XGBoost':
#       model_obj = XGBRegressor(max_depth = 5, subsample = 0.75)
#     model_obj.fit(X_train, y_train)
#     models.append(model_obj)

#     # Get the predictions (OOB for RF)
#     if model == 'RF':
#       pred[:, var] = model_obj.oob_prediction_
#     elif model == 'XGBoost':
#       pred[:, var] = model_obj.predict(X_train)

#   # Unstandardize preds
#   if nn_hyps['standardize'] == True:
#     pred = invert_scaling(pred, scale_output['mu_y'], scale_output['sigma_y'])

#   return {'trained_model': models,
#           'scale_output': scale_output,
#           'standardize': nn_hyps['standardize'],
#           'pred_in': pred,
#           'n_var': n_var}


def predict_ml_model(results, newx):
  scale_output = results['scale_output']
  n_var = results['n_var']

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