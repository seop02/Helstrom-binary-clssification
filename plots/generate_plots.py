import sys
import os
sys.path.append(f"{os.path.dirname(__file__)}/../")
import numpy as np
import plot_funcs
import pandas as pd
from combine import append_datasets, plot_heatmap


from plots import output_path


if __name__ == '__main__':
    
    datasets = ['appendictis', 'echo', 'haberman', 'hepatitis', 'iris','parkinson', 'transfusion', 'wine']

    df = append_datasets(datasets)
    df.to_csv(f'{output_path}/classical_output/f1_final.csv')
    df = pd.read_csv(f'{output_path}/classical_output/f1_final.csv', index_col=0)
    
    plot_heatmap(df, 'all')

    for dataset in datasets:
        score_p = np.load(f'{output_path}/{dataset}_cross_output/{dataset}_score_p.npy')


        score_n = np.load(f'{output_path}/{dataset}_cross_output/{dataset}_score_n.npy')


        score_fidp = np.load(f'{output_path}/{dataset}_cross_output/{dataset}_score_fidp.npy')

        score_fidn = np.load(f'{output_path}/{dataset}_cross_output/{dataset}_score_fidn.npy')
        
        #plot_funcs.raw_scatter_plot(score_p, score_n, score_fidp, score_fidn, dataset)
        plot_funcs.score_plot(score_p, score_n, score_fidp, score_fidn, dataset)
        plot_funcs.f1_plot(dataset)
        
        
        
 
        

    





