import pandas as pd
import numpy as np


def load_data(dataset_name):

    if dataset_name == 'Merge_data_sans_conso':
        data = pd.read_excel('data/Merge_data.xlsx')
        data = data.drop("consommation", axis = 1)
        data["investissement"]=np.log(data["investissement"])
        data["PIB"]=np.log(data["PIB"])
        data["CPI"]=np.log(data["CPI"])
        data["investissement"]=data["investissement"].diff()
        data["PIB"]=data["PIB"].diff()
        data["CPI"]=data["CPI"].diff()
        data["interet"]=data["interet"].diff()
        data["Labour"]=data["Labour"].diff()
        data=data.dropna()
        x_d_all = data[["investissement", "PIB", 'CPI', 'Labour', 'interet']]
        x_d_all.columns = ["investissement", "PIB", 'CPI', 'Labour', 'interet']

    elif dataset_name == 'quarterly':
        data = pd.read_excel('data/Merge_data.xlsx')
        data = data.drop("PIB", axis = 1)
        data["investissement"]=np.log(data["investissement"])
        data["consommation"]=np.log(data["consommation"])
        data["PIB"]=np.log(data["PIB"])
        data["CPI"]=np.log(data["CPI"])
        data["investissement"]=data["investissement"].diff()
        data["consommation"]=data["consommation"].diff()
        data["CPI"]=data["CPI"].diff()
        data["interet"]=data["interet"].diff()
        data["Labour"]=data["Labour"].diff()
        data=data.dropna()
        x_d_all = data[["consommation", 'investissement', 'CPI', 'Labour', 'interet']]
        x_d_all.columns = ["consommation", 'investissement', 'CPI', 'Labour', 'interet']

    else:
        raise ValueError('No such dataset found!')
    
    n_var = len(x_d_all.columns)
    var_names = list(x_d_all.columns)

    print(f'DataLoader: Loaded dataset {dataset_name}')

    return x_d_all, n_var, var_names, None