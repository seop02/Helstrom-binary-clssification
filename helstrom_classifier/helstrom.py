import logging
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold
from torch.nn.functional import normalize
import sys
sys.path.append("./")
from helstrom_classifier import output_path

LOG = logging.getLogger(f"helstrom.{__name__}")


def calculation(a_index, b_index, X_prime, y_train, copies, c):
    a = X_prime[y_train == 0][a_index]
    b = X_prime[y_train == 1][b_index]

    inner_product_ab = torch.matmul(a, b)
    inner_product_ca: torch.Tensor = torch.matmul(c, a)
    inner_product_cb: torch.Tensor = torch.matmul(c, b)              
    overlap_ab = float(inner_product_ab * torch.conj(inner_product_ab))
    overlap_ca = float(inner_product_ca * torch.conj(inner_product_ca))
    overlap_cb = float(inner_product_cb * torch.conj(inner_product_cb))

    lambda_ab = np.sqrt(abs(1 - np.power(overlap_ab, copies)))-1e-15
    contrib_ca = np.power(overlap_ca, copies)
    contrib_cb = np.power(overlap_cb, copies)

    result_ab = 1/lambda_ab * (contrib_ca - contrib_cb)
    result_ab_fid = contrib_ca - contrib_cb
    return result_ab, result_ab_fid

def to_torch(X:torch.Tensor, d_type: torch.dtype):
    m, n = X.shape[0], X.shape[1]
    X_inner = torch.tensor(X, dtype=d_type)

    X_prime = normalize(torch.cat(
        [X_inner, torch.ones(m, dtype=d_type).reshape(-1, 1)], dim=1
    ), p=2, dim=1)
    return X_prime


def helstrom_simulator(X:torch.Tensor, y:torch.Tensor, max_copies: int, d_type: torch.dtype, name: str, start:float, step_size:float):

    if not os.path.exists(f'./output/{name}_cross_output'):
        os.mkdir(f'./output/{name}_cross_output')
        
    if not os.path.exists(f'./figures/{name}_cross_figures'):
        os.mkdir(f'./figures/{name}_cross_figures')

    X_prime = to_torch(X, d_type=d_type)

    f1 = np.zeros(int((max_copies-start)/step_size + 1))
    f1_fid = np.zeros(int((max_copies-start)/step_size + 1))
    score_fidp = []
    score_fidn = []

    score_p = []
    score_n = []

    copies_index = np.linspace(start,max_copies,int((max_copies-start)/step_size + 1))

    n_splits = 5
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=2652124)
    pool = Pool()

    copies = start
    index = 0
    while copies <= max_copies:
        p_cross = []
        n_cross = []

        fidp_cross = []
        fidn_cross = []

        for no_fold, (train_index, test_index) in enumerate(rkf.split(X)):
            X_train = X_prime[train_index]
            y_train = y[train_index]

            X_test = X_prime[test_index]
            y_test = y[test_index]

            result_sum = np.zeros((X_test.shape[0]))
            result_sum_fid = np.zeros((X_test.shape[0]))

            train_arguments = [
                (a_index, b_index, X_train, y_train, copies)
                for a_index in range(X_train[y_train == 0].shape[0])
                for b_index in range(X_train[y_train == 1].shape[0])
            ]

            LOG.debug(X_train.shape)
            LOG.debug(X_test.shape)
            for no_fold in range(X_test.shape[0]):
                LOG.debug(f"Test Data #{no_fold}...")

                c = X_test[no_fold]
                arguments = [t + tuple([c]) for t in train_arguments]

                LOG.debug(f"There are {len(arguments)} of processes to calculate...")
                pool_result = pool.starmap(calculation, arguments)
                result_sum[no_fold] = sum([t[0] for t in pool_result])
                result_sum_fid[no_fold] = sum([t[1] for t in pool_result])
            
            LOG.debug(result_sum)

            normalization = X_train[y_train == 0].shape[0] * X_train[y_train == 1].shape[0]
            score_sum = result_sum / normalization
            score_sum_fid = result_sum_fid / normalization
            
            result_sum = 0.5*(1-np.sign(result_sum))
            result_sum_fid = 0.5*(1-np.sign(result_sum_fid))

            LOG.debug(result_sum)

            f1[index] += f1_score(y_test, result_sum,  average ='binary')
            f1_fid[index] += f1_score(y_test, result_sum_fid,  average ='binary')

            p_cross.append(score_sum[y_test==0])
            n_cross.append(score_sum[y_test==1])

            fidp_cross.append(score_sum_fid[y_test==0])
            fidn_cross.append(score_sum_fid[y_test==1])

            LOG.info(f'Current copy is {copies} current split is {no_fold} current f1 score is {f1[index] / (no_fold + 1)}')
        
        f1[index] = f1[index]/n_splits
        f1_fid[index] = f1_fid[index]/n_splits
        
        score_p.append(p_cross)
        score_n.append(n_cross)
        
        score_fidp.append(fidp_cross)
        score_fidn.append(fidn_cross)


        d = {
            'copies' : copies_index,
            'f1' : f1,
            'f1_fid': f1_fid
        }
        df_f1 = pd.DataFrame(data = d)

        df_p = pd.DataFrame(score_p, index=[copies_index[int(start)-1:(index+1)]])
        df_n = pd.DataFrame(score_n, index=[copies_index[int(start)-1:(index+1)]])
        
        df_fidp = pd.DataFrame(score_fidp, index=[copies_index[int(start)-1:(index+1)]])
        df_fidn = pd.DataFrame(score_fidn, index=[copies_index[int(start)-1:(index+1)]])

        df_f1.to_csv(f'{output_path}/{name}_cross_output/{name}_cross_f1_{start}_{max_copies}_{step_size}.csv') 
        
        np.save(
                f'{output_path}/{name}_cross_output/{name}_cross_scorep_{start}_{max_copies}_{step_size}.npy', 
                df_p
                ) 
        np.save(
                f'{output_path}/{name}_cross_output/{name}_cross_scoren_{start}_{max_copies}_{step_size}.npy', 
                df_n
                ) 
        
        np.save(
                f'{output_path}/{name}_cross_output/{name}_cross_fid_scorep_{start}_{max_copies}_{step_size}.npy', 
                df_fidp
                ) 
        np.save(
                f'{output_path}/{name}_cross_output/{name}_cross_fid_scoren_{start}_{max_copies}_{step_size}.npy', 
                df_fidn
                ) 

        copies = copies + step_size
        index += 1

    pool.close()


def plot_graphs(name, start, max_copies):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    def plot_cross_graphs(fig, ax1, ax2):       
        df = pd.read_csv(f'{output_path}/{name}_cross_output/{name}_cross_data_{start}_{max_copies}.csv')

        ax1.set_xlabel('copies')
        ax1.set_ylabel('Classification score', color='black')
        ax1.plot(df['copies'], df['score_p'], color='tab:blue', label= 'Helstrom Simulation')
        ax1.plot(df['copies'], df['score_n'], color='tab:blue', linestyle = 'dotted')
        ax1.plot(df['copies'], df['score_fidp'], color='tab:red', label= 'Fidelity Simulation', linestyle = 'dashed')
        ax1.plot(df['copies'], df['score_fidn'], color='tab:red', linestyle = 'dashdot')
        ax1.axhline(y=0, color='black', linewidth = 0.8)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc=2)

        if name == 'iris':
            result = []
            for copies in range(1, 7):
                df1 = pd.read_csv(f'output\iris_cross_output\experiment\classification test_1_0 copies_{copies}.csv')
                df1['sign_of_eigenvalues_stabil'] = df1['sign_of_eigenvalues']
                df1.loc[np.abs(df1['eigenvalues']) < 1e-3, 'sign_of_eigenvalues_stabil'] = 0
                result_sum_experiment = np.dot(df1['sign_of_eigenvalues'], df1['overlap'])
            # result_sum_experiment = np.dot(df['sign_of_eigenvalues_stabil'], df['overlap'])
                result_sum_experiment_fid = np.dot(df1['eigenvalues'], df1['overlap'])
                df_copies = pd.DataFrame(
                data=[[copies, result_sum_experiment, result_sum_experiment_fid]],
                columns=['Copies', 'Helstrom exp', 'Fidelity exp']
                )
                result.append(df_copies)
            df2 = pd.concat(result)
            print(df2.head())
            ax1.scatter(df2['Copies'], df2['Helstrom exp'], color='tab:blue', label= 'Helstrom Experiment')
            ax1.scatter(df2['Copies'], df2['Fidelity exp'], color='tab:red', label= 'Fidelity Experiment')
        

        if name == 'aids':
            ax2.set_ylim([0.5,1])
    
        elif name == 'appendictis':
            ax2.set_ylim([0.55,0.7])

        elif name == 'banana':
            ax2.set_ylim([0.5,1])
            
        elif name == 'bcc':
            ax2.set_ylim([0.5,1])
            
        elif name == 'echocardiogram':
            ax2.set_ylim([0.75,0.95])
            
        elif name == 'glass':
            ax2.set_ylim([0.75,0.95])
            
        elif name == 'hd':
            ax2.set_ylim([0.75,0.95])
            
        elif name == 'hepatitis':
            ax2.set_ylim([0.75,0.85])
            
        elif name == 'iris':
            ax2.set_ylim([0.6,1.1])
            
        elif name == 'lupus':
            ax2.set_ylim([0.6,1.1])
            
        elif name == 'parkinson':
            ax2.set_ylim([0.6,0.8])
            
        elif name == 'penguin':
            ax2.set_ylim([0.6,0.8])
            
        elif name == 'wine':
            ax2.set_ylim([0.8,0.9])

        ax2.set_ylabel('f1 score', color='black') 
        ax2.plot(df['copies'], df['f1'], color='darkblue', label='Helstrom Simulation')
        ax2.plot(df['copies'], df['f1_fid'], color='darkred', label='Fidelity Simulation', linestyle = '-.')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc=1)
        fig.tight_layout()  
        plt.savefig(f'./figures/{name}_cross_figures/{name}_cross_f1_{max_copies}.png')
        plt.show()

    return plot_cross_graphs(fig, ax1, ax2)

def find_max_f1(names, start, max_copies): 
    max_f1 = np.zeros(len(names))
    max_f1_fid = np.zeros(len(names))
    maxindex_f1 = np.zeros(len(names))
    maxindex_f1_fid = np.zeros(len(names))
    for i in range(len(names)):
        if names[i] == 'wine':
            df = pd.read_csv(f'{output_path}/{names[i]}_cross_output/{names[i]}_cross_data_{start}_300.csv')
        else:
            df = pd.read_csv(f'{output_path}/{names[i]}_cross_output/{names[i]}_cross_data_{start}_{max_copies}.csv')
        f1 = df['f1'].values
        f1_fid = df['f1_fid'].values

        max_f1[i] = np.amax(f1)
        maxindex_f1[i] = np.argmax(f1)+1

        max_f1_fid[i] = np.amax(f1_fid)
        maxindex_f1_fid[i] = np.argmax(f1_fid)+1

    d = {'f1_hel': max_f1, 'copies_hel': maxindex_f1, 'f1_fid': max_f1_fid, 'copies_fid': maxindex_f1_fid}
    df1 = pd.DataFrame(data = d, index = names)
    df1.to_csv(f'{output_path}/classical_output/cross_data.csv')

    return df1
