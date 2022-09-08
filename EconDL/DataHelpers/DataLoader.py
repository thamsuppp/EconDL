import pandas as pd
import numpy as np

# @TODO: Add functionality to output the dates (in the future for evaluation)

def load_data(dataset_name):

    if dataset_name == 'monthly':
        data = pd.read_csv('data/monthlyData.csv')
        data = data.drop('Unnamed: 0', axis = 1)

        data['L0_HOUST'] = data['L0_HOUST'].diff()
        data = data.dropna()
    
        x_d_all = data[['L0_OILPRICEx', 'L0_EXUSUKx', 'L0_S.P.500', 'L0_TB3MS', 'L_0y', 'L0_UNRATE', 'L0_HOUST']]
        x_d_all.columns = ['oil', 'Ex', 'SPY', 'DGS3', 'inf', 'unrate', 'house_starts']
        exog_data = data[[e for e in data.columns if e not in ['L0_OILPRICEx', 'L0_EXUSUKx', 'L0_S.P.500', 'L0_TB3MS', 'L_0y', 'L0_UNRATE', 'L0_HOUST']]]
        
    elif dataset_name == 'quarterly':
        data = pd.read_csv('data/monthlyData.csv')
        data['quarter'] = ((data['trend'] ) / 3).astype(int)
        data = data.groupby('quarter').mean().reset_index()
        data['L0_HOUST'] = data['L0_HOUST'].diff()
        data = data.dropna()

        x_d_all = data[['L0_OILPRICEx', 'L0_EXUSUKx', 'L0_S.P.500', 'L0_TB3MS', 'L_0y', 'L0_UNRATE', 'L0_HOUST']]
        x_d_all.columns = ['oil', 'Ex', 'SPY', 'DGS3', 'inf', 'unrate', 'house_starts']
        exog_data = data[[e for e in data.columns if e not in ['L0_OILPRICEx', 'L0_EXUSUKx', 'L0_S.P.500', 'L0_TB3MS', 'L_0y', 'L0_UNRATE', 'L0_HOUST']]]

    elif dataset_name == 'varctic':
        data = pd.read_csv('data/VARCTIC8.csv')
        data = data.dropna()
        x_d_all = data[['CO2_MaunaLoa', 'TCC', 'PR', 'AT', 'SST', 'SIE', 'SIT', 'Albedo']]
        x_d_all.columns = ['CO2_MaunaLoa', 'TCC', 'PR', 'AT', 'SST', 'SIE', 'SIT', 'Albedo']
        exog_data = None
    elif dataset_name == 'financial':
        data = pd.read_csv('data/ryan_data_h1.csv')
        # Shift all the Y values up by 1 period (so that yesterday's predictors predict today's Y)
        data[['index', 'Y_sp', 'Y_nas', 'Y_vix', 'Y_dj']] = data[['index', 'Y_sp', 'Y_nas', 'Y_vix', 'Y_dj']].shift(-1)
        data = data.dropna()
        x_d_all = data[['Y_sp', 'Y_nas', 'Y_vix', 'Y_dj']]
        x_d_all.columns = ['S&P', 'NASDAQ', 'VIX', 'DJIA']
        exog_data = data[[e for e in data.columns if e not in ['index', 'Y_sp', 'Y_nas', 'Y_vix', 'Y_dj']]]

    elif dataset_name == 'financial_3var':
        data = pd.read_csv('data/ryan_data_h1.csv')
        # Shift all the Y values up by 1 period (so that yesterday's predictors predict today's Y)
        data[['index', 'Y_sp', 'Y_nas', 'Y_dj']] = data[['index', 'Y_sp', 'Y_nas', 'Y_dj']].shift(-1)
        data = data.dropna()
        x_d_all = data[['Y_sp', 'Y_nas', 'Y_dj']]
        x_d_all.columns = ['S&P', 'NASDAQ', 'DJIA']
        exog_data = data[[e for e in data.columns if e not in ['index', 'Y_sp', 'Y_nas', 'Y_dj']]]

    else:
        raise ValueError('No such dataset found!')
    
    n_var = len(x_d_all.columns)
    var_names = list(x_d_all.columns)

    print(f'DataLoader: Loaded dataset {dataset_name}')

    return x_d_all, n_var, var_names, exog_data