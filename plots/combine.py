import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(f"{os.path.dirname(__file__)}/../")

from plots import output_path, figures_path

def plot_heatmap(df, name):
    x_label = [
        'LR', 'DT', 'NC', 'LDA', 'QDA', 'BNB', 'GNB', 'SVM', 'ADA' ,'RF', 'KNN', 'XG', 'CAT','Hel', 'Fid'
        ]
    df = df.transpose()
    fig, ax = plt.subplots(figsize=(20,10))  
    sns.set(font_scale=2.0)
    sns.heatmap(df, annot=True, cmap = 'Blues', annot_kws={'size': 22}, ax=ax)
    ax.set_xticklabels(labels=x_label,fontsize=22)
    ax.set_yticklabels(labels=df.index.tolist(), fontsize=18)
    plt.title('Cross validated f1 score for all classifiers')
    plt.tight_layout()
    plt.savefig(f'{figures_path}/f1_for_all.pdf')
    plt.show()

def helstrom_combine(datasets):
    f1_hel = np.zeros((len(datasets)))
    f1_fid = np.zeros((len(datasets)))
    for i,dataset in enumerate(datasets):
        if dataset == 'wine' or dataset == 'haberman' or dataset == 'transfusion':
            df = pd.read_csv(f'{output_path}/{dataset}_cross_output/{dataset}_cross_f1_0.25_300.0_0.25.csv', index_col=0)
            f1_hel[i] = df['f1'].max()
            f1_fid[i] = df['f1_fid'].max()
        else:
            df = pd.read_csv(f'{output_path}/{dataset}_cross_output/{dataset}_cross_f1_0.25_100.0_0.25.csv', index_col=0)
            f1_hel[i] = df['f1'].max()
            f1_fid[i] = df['f1_fid'].max()
        
    return f1_hel, f1_fid

def append_datasets(datasets):
    for data in datasets:
        if data == datasets[0]:
            df = pd.read_csv(f'{output_path}/classical_output/f1_{data}.csv', index_col=[0])
        else:
            df1 = pd.read_csv(f'{output_path}/classical_output/f1_{data}.csv', index_col=[0])
            df = pd.concat([df, df1['f1_score']], axis=1, ignore_index=True)
    df.set_index(df.columns[0], inplace=True)
    df.to_csv(f'{output_path}/classical_output/f1_classic.csv')
    df.columns = datasets

    f1_hel, f1_fid = helstrom_combine(datasets)
    new = pd.DataFrame([f1_hel, f1_fid], columns=df.columns, index=['f1_hel', 'f1_fid'])
    df_final=pd.concat([df, new], axis=0)

    max_values = df_final.head(4).max()
    max_row_df = pd.DataFrame([max_values], index=['LogisticR'])
    df_final.iloc[:4, :] = max_row_df.values
    df_final = df_final.drop(['LogisticR_liblinear', 'LogisticR_newtonc', 'LogisticR_newtonch'])
    df_final = df_final.rename(index={'LogisticR_lbfgs': 'LogisticR'})

    df_final.to_csv(f'{output_path}/classical_output/f1_final.csv')
    return df_final
