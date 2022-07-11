import pandas as pd
import numpy as np

def load_data(experiment_params):

    if experiment_params['dataset'] in ['monthly', 'quarterly']:
        filename = 'monthlyData.csv'
    elif experiment_params['dataset'] == 'varctic':
        filename = 'VARCTIC8.csv'
    elif experiment_params['dataset'] == 'financial':
        filename = 'ryan_data_h1.csv'
    else:
        raise ValueError('No such dataset found!')
    
    path = f'../data/{filename}'
    data = pd.read_csv(path)
    return data