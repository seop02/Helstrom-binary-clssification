import os.path

import numpy as np
import pandas as pd
import torch
from pmlb import fetch_data
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from torch.nn.functional import normalize
import sys
sys.path.append("./")
from helstrom_classifier import data_set_folder

def load_datasets(dataset):
    if dataset == 'aids':
        X, y = fetch_data('analcatdata_aids', return_X_y=True)
        return X, y
    
    elif dataset == 'appendictis':
        X, y = fetch_data('appendicitis', return_X_y=True)
        return X, y
    
    elif dataset == 'banana':
        X, y = fetch_data('banana', return_X_y=True)
        return X, y
    
    elif dataset == 'bcc':
        X, y = load_breast_cancer(return_X_y=True)
        return X, y
    
    elif dataset == 'echocardiogram':
        echocardio = pd.read_csv(f'{data_set_folder}/echocardiogram.data', sep=',', on_bad_lines='skip')
        echocardio = echocardio.drop(columns=['name', 'group']) #irrelevant data
        echocardio = echocardio.replace("?", np.nan)
        echocardio = echocardio.dropna().reset_index(drop=True)
        X = np.array(echocardio.iloc[:, 0:10].values, dtype=np.float64)
        y = np.array(echocardio.iloc[:, 10].values, dtype=np.float64)
        
        X = torch.from_numpy(X)
        return X, y
    
    elif dataset == 'glass':
        X, y = fetch_data('glass2', return_X_y=True)
        return X, y
    
    elif dataset == 'hd':
        data = pd.read_csv(f'{data_set_folder}/heart_disease.csv')
        y = data['label'].values
        X = data.drop('label', axis = 1).values
        X = X.astype(float)
        return X, y
    
    elif dataset == 'hepatitis':
        X, y = fetch_data('hepatitis', return_X_y=True)
        y-=1
        return X, y
    
    elif dataset == 'ionosphere':
        names = np.zeros(35)
        for i in range(35):
            names[i] = i

        ion = pd.read_csv(f'{data_set_folder}/ionosphere.data', sep=',', names = names)

        ion=ion.dropna()
        ion=ion.dropna(axis=0)
        ion=ion.dropna().reset_index(drop=True)

        labels = np.zeros(351)
        for i in range(351):
            if ion[34.0][i] == 'g':
                labels[i] = 0
            else:
                labels[i] = 1

        ion['labels'] = labels
        X = ion.iloc[:, 0:33].values
        y = ion.iloc[:, 35].values
        return X, y
    
    elif dataset == 'iris':
        X,y = load_iris(return_X_y=True)
        y1 =  y[y != 2]
        X = X[y != 2]
        return X, y1
    
    elif dataset == 'lupus':
        X, y = fetch_data('lupus', return_X_y=True)
        return X, y
        
    
    elif dataset == 'parkinson':
        data = pd.read_csv(f'{data_set_folder}/parkinsons.data')
        data=data.dropna()
        data=data.dropna(axis=0)
        data=data.dropna().reset_index(drop=True)

        y = data['status'].values
        X = data.drop('status',axis=1)
        X = X.drop('name', axis = 1).values
        return X, y
    
    elif dataset == 'penguin':
        X, y = fetch_data('penguins', return_X_y=True)
        y1 =  y[y != 2]
        X = X[y != 2]
        return X, y1
    
    elif dataset == 'wine':
        X, y = load_wine(return_X_y=True)
        y1 =  y[y != 2]
        X = X[y != 2]
        return X, y1
    
    elif dataset == 'yeast':
        df = pd.read_csv(f'{data_set_folder}/yeast.data', header=None, delimiter=r'\s+')
        y_raw = df[9].values
        y = np.zeros((len(y_raw)))
        for i,j in enumerate(y_raw):
            if j == 'ME2':
                y[i] = 1
        X = np.array(df.iloc[:, 1:9].values, dtype=np.float64)
        return X, y
    
    elif dataset == 'haberman':
        df = pd.read_csv(f'{data_set_folder}/haberman.data', header=None, delimiter=',')
        y_raw = df[3].values
        y = y_raw -1
        X = np.array(df.iloc[:, 0:3].values, dtype=np.float64)
        return X, y
    
    elif dataset == 'transfusion':
        df = pd.read_csv(f'{data_set_folder}/transfusion.data', delimiter=",")
        y = df['4'].values
        X = np.array(df.iloc[:, 0:4].values, dtype=np.float64)
        return X, y

    
    else:
        print('the data set does not exist')

def to_torch(X:torch.Tensor, d_type: torch.dtype):
    m, n = X.shape[0], X.shape[1]
    X_inner = torch.tensor(X, dtype=d_type)

    X_prime = normalize(torch.cat(
        [X_inner, torch.ones(m, dtype=d_type).reshape(-1, 1)], dim=1
    ), p=2, dim=1)
    X_prime = X_prime.numpy()
    return X_prime
