import numpy as np
import os

folder_path = 'results/16jul_diff_order_original/'
out_folder_path = 'results/16jul_diff_order/inverted/'

M_varnn = 6

# We want DGS3-inf-unrate
invert_order_dict = {
  0: [0, 1, 2],
  1: [0, 2, 1],
  2: [1, 0, 2],
  3: [2, 0, 1],
  4: [1, 2, 0],
  5: [2, 1, 0]
}

for i in range(M_varnn):

  # Invert the results
  load_file = f'{folder_path}/params_{i}_compiled.npz'

  print(f'Evaluation load_results(): load_file: {load_file}')

  results = np.load(load_file, allow_pickle = True)['results'].item()
  params = results['params']

  BETAS = results['betas']
  BETAS_IN = results['betas_in']
  SIGMAS = results['sigmas']
  SIGMAS_IN = results['sigmas_in']
  PRECISION = results['precision']
  PRECISION_IN = results['precision_in']
  CHOLESKY = results['cholesky']
  CHOLESKY_IN = results['cholesky_in']
  PREDS = results['train_preds']
  PREDS_TEST = results['test_preds']
  Y_train = results['y']
  Y_test = results['y_test']

  # Reverse all
  BETAS = BETAS[:, [e+1 for e in invert_order_dict[i]], :, invert_order_dict[i], :]
  BETAS_IN = BETAS_IN[:, [e+1 for e in invert_order_dict[i]], :, invert_order_dict[i], :]
  SIGMAS = SIGMAS[:, invert_order_dict[i], invert_order_dict[i], :]
  SIGMAS_IN = SIGMAS_IN[:, invert_order_dict[i], invert_order_dict[i], :]
  PRECISION = PRECISION[:, invert_order_dict[i], invert_order_dict[i], :]
  PRECISION_IN = PRECISION_IN[:, invert_order_dict[i], invert_order_dict[i], :]
  CHOLESKY = CHOLESKY[:, invert_order_dict[i], invert_order_dict[i], :, :]
  CHOLESKY_IN = CHOLESKY_IN[:, invert_order_dict[i], invert_order_dict[i], :, :]
  PREDS = PREDS[:, :, invert_order_dict[i]]
  PREDS_TEST = PREDS_TEST[:, :, invert_order_dict[i]]
  params['var_names'] = ["DGS3", "inf", "unrate"]

  results_out = {
    'betas': BETAS,
    'betas_in': BETAS_IN,
    'sigmas': SIGMAS,
    'sigmas_in': SIGMAS_IN,
    'precision': PRECISION,
    'precision_in': PRECISION_IN,
    'cholesky': CHOLESKY,
    'cholesky_in': CHOLESKY_IN,
    'train_preds': PREDS,
    'test_preds': PREDS_TEST,
    'y': Y_train,
    'y_test': Y_test,
    'params': params
  }

  with open(f'{out_folder_path}/params_{i}_compiled.npz', 'wb') as f:
    np.savez(f, results = results_out)

  print(f'Experiment {i}: Inverted results')

  # Invert fcast
  load_file = f'{folder_path}/fcast_{i}_compiled.npz'
  out = np.load(load_file)
  fcast = out['fcast']
  fcast = fcast[:, :, invert_order_dict[i], invert_order_dict[i], :]

  with open(f'{out_folder_path}/fcast_{i}_compiled.npz', 'wb') as f:
    np.savez(f, fcast = fcast)

  print(f'Experiment {i}: Inverted fcast')

  # Invert fcast compiled
  load_file = f'{folder_path}/multi_fcast_{i}_compiled.npz'
  out = np.load(load_file)
  fcast = out['fcast']
  fcast = fcast[:, invert_order_dict[i], :, :, :]

  with open(f'{out_folder_path}/multi_fcast_{i}_compiled.npz', 'wb') as f:
    np.savez(f, fcast = fcast)




  print(f'Experiment {i}: Inverted multi fcast')

  