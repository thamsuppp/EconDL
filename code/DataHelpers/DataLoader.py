import pandas as pd
import numpy as np

# @TODO: Add functionality to output the dates (in the future for evaluation)

def load_data(dataset_name):

    if dataset_name == 'monthly':
        data = pd.read_csv('../data/monthlyData.csv')
        x_d_all = data[['L0_OILPRICEx', 'L0_EXUSUKx', 'L0_S.P.500', 'L0_TB3MS', 'L_0y', 'L0_UNRATE', 'L0_HOUST']]
        x_d_all['L0_HOUST'] = x_d_all['L0_HOUST'].diff()
        x_d_all = x_d_all.dropna()
        x_d_all.columns = ['oil', 'Ex', 'SPY', 'DGS3', 'inf', 'unrate', 'house_starts']
    elif dataset_name == 'quarterly':
        data = pd.read_csv('../data/monthlyData.csv')
        data['quarter'] = ((data['trend'] ) / 3).astype(int)
        data = data.groupby('quarter').mean().reset_index()
        data['L0_HOUST'] = data['L0_HOUST'].diff()
        data = data.dropna()
        x_d_all = data[['L0_OILPRICEx', 'L0_EXUSUKx', 'L0_S.P.500', 'L0_TB3MS', 'L_0y', 'L0_UNRATE', 'L0_HOUST']]
        x_d_all.columns = ['oil', 'Ex', 'SPY', 'DGS3', 'inf', 'unrate', 'house_starts']
    elif dataset_name == 'varctic':
        data = pd.read_csv('../data/VARCTIC8.csv')
        data = data.dropna()
        x_d_all = data[['CO2_MaunaLoa', 'TCC', 'PR', 'AT', 'SST', 'SIE', 'SIT', 'Albedo']]
        x_d_all.columns = ['CO2_MaunaLoa', 'TCC', 'PR', 'AT', 'SST', 'SIE', 'SIT', 'Albedo']
    elif dataset_name == 'financial':
        data = pd.read_csv('../data/ryan_data_h1.csv')
        data = data.dropna()
        x_d_all = data[['Y_sp', 'Y_nas', 'Y_vix', 'Y_dj']]
        x_d_all.columns = ['S&P', 'NASDAQ', 'VIX', 'DJIA']
    else:
        raise ValueError('No such dataset found!')
    
    n_var = len(x_d_all.columns)
    var_names = list(x_d_all.columns)

    print(f'DataLoader: Loaded dataset {dataset_name}')

    return x_d_all, n_var, var_names