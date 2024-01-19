import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib as mpl

import sys
import os
sys.path.append(f"{os.path.dirname(__file__)}/../")

from plots import figures_path, output_path


def box_plot(score_p, score_n, score_fidp, score_fidn, name, max_copies):
    pos, neg = True, True
    # pos, neg = True, False
    # pos, neg = False, True

    n_th_element = 1
    score_p = score_p.copy()[0::n_th_element]
    score_n = score_n.copy()[0::n_th_element]
    score_fidp = score_fidp.copy()[0::n_th_element]
    score_fidn = score_fidn.copy()[0::n_th_element]

    if name == 'wine':
        # copies = np.linspace(0.25, 300, 1200)
        copies = np.linspace(0.25, 300, int(1200/n_th_element))
    else:
        # copies = np.linspace(0.25, 100, 400)
        copies = np.linspace(0.25, 100, int(400/n_th_element))

    fig: Figure = plt.figure(figsize =(20, 10))
    # Creating plot
    c = [0,0,0.7,0.5]
    n = [0.7,0,0,0.5]
    if pos:
        plt.plot(copies, np.mean(score_p, 1),  color='blue', label='Helstrom positive class')
        plt.boxplot(
            score_p.T, positions=copies, labels=[""]*len(copies), manage_ticks=False,
            notch=False, patch_artist=True,
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color='navy')
        )
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Box plot for Helstrom simulation of {name}")
    plt.savefig(f'{figures_path}/box_figures/{name}_helstrom_p.png')
    plt.savefig(f'{figures_path}/box_figures/{name}_helstrom_p.pdf')
    plt.show()
    fig: Figure = plt.figure(figsize =(20, 10))
    if neg:
        plt.plot(copies, np.mean(score_n, 1),  color='red', label='Helstrom negative class')
        plt.boxplot(
            score_n.T, positions=copies, labels=[""]*len(copies), manage_ticks=False,
            notch=False, patch_artist=True,
            boxprops=dict(facecolor= n, color=n),
            capprops=dict(color=n),
            whiskerprops=dict(color=n),
            flierprops=dict(color=n, markeredgecolor=n),
            medianprops=dict(color='darkred')
        )
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Box plot for Helstrom simulation of {name}")
    plt.savefig(f'{figures_path}/{name}_cross_figures/{name}_helstrom_box_{max_copies}.png')
    plt.savefig(f'{figures_path}/{name}_cross_figures/{name}_helstrom_box_{max_copies}.pdf')
    plt.show()
    
    fig: Figure = plt.figure(figsize =(20, 10))
    # Creating plot
    if pos:
        plt.plot(copies, np.mean(score_fidp, 1),  color='blue', label='Fidelity positive class')
        plt.boxplot(
            score_fidp.T, positions=copies, labels=[""]*len(copies), manage_ticks=False,
            notch=False, patch_artist=True,
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color='navy')
        )
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Box plot for Fidelity simulation of {name}")
    plt.savefig(f'{figures_path}/box_figures/{name}_fidelity_p.png')
    plt.savefig(f'{figures_path}/box_figures/{name}_fidelity_p.pdf')
    plt.show()

    fig: Figure = plt.figure(figsize =(20, 10))
    if neg:
        plt.plot(copies, np.mean(score_fidn, 1),  color='red', label='Fidelity negative class')
        plt.boxplot(
            score_fidn.T, positions=copies, labels=[""]*len(copies), manage_ticks=False,
            notch=False, patch_artist=True,
            boxprops=dict(facecolor= n, color=n),
            capprops=dict(color=n),
            whiskerprops=dict(color=n),
            flierprops=dict(color=n, markeredgecolor=n),
            medianprops=dict(color='darkred')
        )
    #plt.xticks(ticks=x, labels=x)
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Box plot for Fidelity simulation of {name}")
    plt.savefig(f'{figures_path}/{name}_cross_figures/{name}_fidelity_box_{max_copies}.png')
    plt.savefig(f'{figures_path}/{name}_cross_figures/{name}_fidelity_box_{max_copies}.pdf')
    plt.show()


def raw_scatter_plot(score_p, score_n, score_fidp, score_fidn, name):
    pos, neg = True, True
    # pos, neg = True, False
    # pos, neg = False, True

    n_th_element = 1
    score_p = score_p.copy()[0::n_th_element]
    score_n = score_n.copy()[0::n_th_element]
    score_fidp = score_fidp.copy()[0::n_th_element]
    score_fidn = score_fidn.copy()[0::n_th_element]

    if name == 'wine':
        # copies = np.linspace(0.25, 300, 1200)
        copies = np.linspace(0.25, 300, int(1200 / n_th_element))
    elif name == 'haberman' or name == 'transfusion':
        # copies = np.linspace(0.25, 300, 1200)
        copies = np.linspace(0.25, 300, int(1200 / n_th_element)-3)
    else:
        # copies = np.linspace(0.25, 100, 400)
        copies = np.linspace(0.25, 100, int(400 / n_th_element))

    fig: Figure = plt.figure(figsize=(20, 10))
    # Creating plot
    c = [0, 0, 0.7, 0.5]
    n = [0.7, 0, 0, 0.5]
    if pos:
        plt.plot(copies, np.mean(score_p, 1), color='blue', label='Helstrom positive class')

        cops = np.asarray([[c] * score_p.shape[1] for c in copies])
        plt.scatter(
            x=cops, y=score_p, alpha=.02, color='navy', marker="o"
        )
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Scatter plot for Helstrom simulation of {name}")
    plt.savefig(f'{figures_path}/box_figures/{name}_helstrom_p.png')
    plt.savefig(f'{figures_path}/box_figures/{name}_helstrom_p.pdf')
    plt.show()
    fig: Figure = plt.figure(figsize=(20, 10))
    if neg:
        plt.plot(copies, np.mean(score_n, 1), color='red', label='Helstrom negative class')
        cops = np.asarray([[c] * score_n.shape[1] for c in copies])
        plt.scatter(
            x=cops, y=score_n, alpha=.02, color='darkred', marker="o"
        )
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Scatter plot for Helstrom simulation of {name}")
    plt.savefig(f'{figures_path}/box_figures/{name}_helstrom_n.png')
    plt.savefig(f'{figures_path}/box_figures/{name}_helstrom_n.pdf')
    plt.show()


def std_plot(score_p, score_n, score_fidp, score_fidn, name, max_copies):
    if name == 'wine':
        copies = np.linspace(0.25, 300, 1200) 
    else:
        copies = np.linspace(0.25, 100, 400)
    
    fig: Figure = plt.figure(figsize =(10, 8))
    ax: Axes = fig.subplots()
    
    plt.plot(copies, np.mean(score_p, 1),  color='blue', label='Helstrom positive class')
    plt.plot(copies, np.mean(score_n, 1),  color='red', label='Helstrom negative class')

    p_error = np.std(score_p, 1)
    n_error = np.std(score_n, 1)

    plt.fill_between(copies, np.mean(score_p, 1)-p_error, np.mean(score_p, 1)+p_error, 
                     alpha=0.3, edgecolor='darkblue', facecolor='blue')
    plt.fill_between(copies, np.mean(score_n, 1) - n_error, np.mean(score_n, 1)+n_error,
                     alpha=0.3, edgecolor='darkred', facecolor='red', linestyle = 'dotted', antialiased=True)
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Box plot for the Helstrom simulation of {name}")
    plt.savefig(f'{figures_path}/std_plot/{name}_hel_std_{max_copies}.png')
    plt.show()

    fig: Figure = plt.figure(figsize =(10, 8))
    ax: Axes = fig.subplots()
    
    plt.plot(copies, np.mean(score_fidp, 1),  color='blue', label='Fidelity positive class')
    plt.plot(copies, np.mean(score_fidn, 1),  color='red', label='Fidelity negative class')

    p_error = np.std(score_fidp, 1)
    n_error = np.std(score_fidn, 1)

    plt.fill_between(copies, np.mean(score_fidp, 1)-p_error, np.mean(score_fidp, 1)+p_error, 
                     alpha=0.3, edgecolor='darkblue', facecolor='blue')
    plt.fill_between(copies, np.mean(score_fidn, 1)-n_error, np.mean(score_fidn, 1)+n_error,
                     alpha=0.3, edgecolor='darkred', facecolor='red', linestyle = 'dotted', antialiased=True)
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Box plot for fidelity simulation of {name}")
    plt.savefig(f'{figures_path}/std_plot/{name}_fidelity_std_{max_copies}.png')
    plt.show()

    fig: Figure = plt.figure(figsize =(10, 8))
    ax: Axes = fig.subplots()
    
    plt.plot(copies, np.mean(score_p, 1),  color='blue', label='Helstrom positive class')
    plt.plot(copies, np.mean(score_fidp, 1),  color='red', label='Fidelity negative class', linestyle='dotted')

    p_error = np.std(score_p, 1)
    fidp_error = np.std(score_fidp, 1)

    plt.fill_between(copies, np.mean(score_p, 1)-p_error, np.mean(score_p, 1)+p_error, 
                     alpha=0.3, edgecolor='darkblue', facecolor='blue')
    plt.fill_between(copies, np.mean(score_fidp, 1)-fidp_error, np.mean(score_fidp, 1)+fidp_error,
                     alpha=0.3, edgecolor='darkred', facecolor='red', linestyle = 'dotted', antialiased=True)
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"Box plot for fidelity simulation of {name}")
    plt.savefig(f'{figures_path}/std_plot/{name}_comparison_std_{max_copies}.png')
    plt.show()



def f1_plot(dataset):
    n_th_element = 1

    if dataset == 'wine':
        # copies = np.linspace(0.25, 300, 1200)
        copies = np.linspace(0.25, 300, int(1200 / n_th_element))
        df = pd.read_csv(f'output/{dataset}_cross_output/{dataset}_cross_f1_0.25_300.0_0.25.csv')

    elif dataset == 'haberman' or dataset == 'transfusion':
        # copies = np.linspace(0.25, 300, 1200)
        copies = np.linspace(0.25, 300, int(1200 / n_th_element)-3)
        df = pd.read_csv(f'output/{dataset}_cross_output/{dataset}_cross_f1_0.25_300.0_0.25.csv')
    else:
        # copies = np.linspace(0.25, 100, 400)
        copies = np.linspace(0.25, 100, int(400 / n_th_element))
        df = pd.read_csv(f'output/{dataset}_cross_output/{dataset}_cross_f1_0.25_100.0_0.25.csv')

    fig: Figure = plt.figure(figsize =(10, 8))
    ax: Axes = fig.subplots()
    #mpl.style.use('classic')
    plt.rcParams.update({'font.size': 14})
    plt.plot(copies, df['f1'],  color='blue', label='Helstrom simulation')
    plt.plot(copies, df['f1_fid'],  color='red', label='Fidelity simulation', linestyle = 'dotted')
    plt.xlabel('copies')
    plt.ylabel('F1 score')
    hel_copies = (np.argmax(df['f1'].values)+1)/4
    hel_max = np.amax(df['f1'].values)
    fid_copies = (np.argmax(df['f1_fid'].values)+1)/4
    fid_max = np.amax(df['f1_fid'].values)
    if dataset=='appendictis':
        plt.annotate(f'helstrom:({hel_copies},{hel_max:.3f})',(hel_copies, hel_max), fontsize = 14, color = 'blue', 
                    xytext = (hel_copies-15, hel_max+0.0025),arrowprops=dict(color='blue',arrowstyle="-|>"))
        plt.annotate(f'fidelity:({fid_copies},{fid_max:.3f})',(fid_copies, fid_max), fontsize = 14, color = 'red', 
                    xytext = (fid_copies+15, fid_max-0.001),arrowprops=dict(color='red',arrowstyle="-|>"))
    elif dataset=='wine':
        plt.annotate(f'helstrom:({hel_copies},{hel_max:.3f})',(hel_copies, hel_max), fontsize = 14, color = 'blue',
                    xytext = (hel_copies-100, hel_max-0.002),arrowprops=dict(color='blue',arrowstyle="-|>"))
        plt.annotate(f'fidelity:({fid_copies},{fid_max:.3f})',(fid_copies, fid_max), fontsize = 14, color = 'red', 
                    xytext = (fid_copies-40, fid_max+0.002),arrowprops=dict(color='red',arrowstyle="-|>"))
    elif dataset=='hepatitis':
        plt.annotate(f'helstrom:({hel_copies},{hel_max:.3f})',(hel_copies, hel_max), fontsize = 14, color = 'blue', 
                    xytext = (hel_copies-30, hel_max-0.003),arrowprops=dict(color='blue',arrowstyle="-|>"))
        plt.annotate(f'fidelity:({fid_copies},{fid_max:.3f})',(fid_copies, fid_max), fontsize = 14, color = 'red', 
                    xytext = (fid_copies-30, fid_max),arrowprops=dict(color='red',arrowstyle="-|>"))
    elif dataset=='echo':
        plt.annotate(f'helstrom:({hel_copies},{hel_max:.3f})',(hel_copies, hel_max), fontsize = 14, color = 'blue', 
                    xytext = (hel_copies-35, hel_max-0.005),arrowprops=dict(color='blue',arrowstyle="-|>"))
        plt.annotate(f'fidelity:({fid_copies},{fid_max:.3f})',(fid_copies, fid_max), fontsize = 14, color = 'red', 
                    xytext = (fid_copies-30, fid_max+0.002),arrowprops=dict(color='red',arrowstyle="-|>"))
    elif dataset=='parkinson':
        plt.annotate(f'helstrom:({hel_copies},{hel_max:.3f})',(hel_copies, hel_max), fontsize = 14, color = 'blue', 
                    xytext = (hel_copies+20, hel_max-0.005),arrowprops=dict(color='blue',arrowstyle="-|>"))
        plt.annotate(f'fidelity:({fid_copies},{fid_max:.3f})',(fid_copies, fid_max), fontsize = 14, color = 'red', 
                    xytext = (fid_copies+20, fid_max+0.003),arrowprops=dict(color='red',arrowstyle="-|>"))
    elif dataset=='haberman':
        plt.annotate(f'helstrom:({hel_copies},{hel_max:.3f})',(hel_copies, hel_max), fontsize = 14, color = 'blue', 
                    xytext = (hel_copies+20, hel_max-0.005),arrowprops=dict(color='blue',arrowstyle="-|>"))
        plt.annotate(f'fidelity:({fid_copies},{fid_max:.3f})',(fid_copies, fid_max), fontsize = 14, color = 'red', 
                    xytext = (fid_copies, fid_max+0.01),arrowprops=dict(color='red',arrowstyle="-|>"))
    elif dataset=='transfusion':
        plt.annotate(f'helstrom:({hel_copies},{hel_max:.3f})',(hel_copies, hel_max), fontsize = 14, color = 'blue', 
                    xytext = (hel_copies+20, hel_max-0.005),arrowprops=dict(color='blue',arrowstyle="-|>"))
        plt.annotate(f'fidelity:({fid_copies},{fid_max:.3f})',(fid_copies, fid_max), fontsize = 14, color = 'red', 
                    xytext = (fid_copies-100, fid_max+0.01),arrowprops=dict(color='red',arrowstyle="-|>"))
    plt.legend()
    plt.title(f"F1 scores of {dataset}")
    plt.savefig(f'{figures_path}/f1_figures/{dataset}_f1_final.png')
    plt.savefig(f'{figures_path}/f1_figures/{dataset}_f1_final.pdf')
    plt.show()


def find_max_f1(names): 
    max_f1 = np.zeros(len(names))
    max_f1_fid = np.zeros(len(names))
    maxindex_f1 = np.zeros(len(names))
    maxindex_f1_fid = np.zeros(len(names))
    for i in range(len(names)):
        if names[i] == 'wine':
            df = pd.read_csv(f'{output_path}/{names[i]}_cross_output/{names[i]}_cross_f1_0.25_300.0.csv')
        else:
            df = pd.read_csv(f'{output_path}/{names[i]}_cross_output/{names[i]}_cross_f1_0.25_100.0.csv')
        f1 = df['f1'].values
        f1_fid = df['f1_fid'].values

        max_f1[i] = np.amax(f1)
        maxindex_f1[i] = (np.argmax(f1)+1)/4

        max_f1_fid[i] = np.amax(f1_fid)
        maxindex_f1_fid[i] = (np.argmax(f1_fid)+1)/4

    d = {'f1_hel': max_f1, 'copies_hel': maxindex_f1, 'f1_fid': max_f1_fid, 'copies_fid': maxindex_f1_fid}
    df1 = pd.DataFrame(data = d, index = names)
    df1.to_csv(f'{output_path}/classical_output/cross_data_final.csv')

    return df1


def score_plot(score_p, score_n, score_fidp, score_fidn, dataset):
    n_th_element = 1

    if dataset == 'wine':
        # copies = np.linspace(0.25, 300, 1200)
        copies = np.linspace(0.25, 300, int(1200 / n_th_element))
    elif dataset == 'haberman' or dataset == 'transfusion':
        # copies = np.linspace(0.25, 300, 1200)
        copies = np.linspace(0.25, 300, int(1200 / n_th_element)-3)
    else:
        # copies = np.linspace(0.25, 100, 400)
        copies = np.linspace(0.25, 100, int(400 / n_th_element))
    
    fig: Figure = plt.figure(figsize =(10, 8))
    ax: Axes = fig.subplots()
    mpl.style.use('classic')

    
    plt.plot(copies, np.mean(score_p, 1),  color='blue', label='Helstrom positive', linestyle = 'solid')
    plt.plot(copies, np.mean(score_n, 1),  color='blue', label='Helstrom negative', linestyle = 'dotted')
    plt.plot(copies, np.mean(score_fidp, 1),  color='red', label='Fidelity positive', linestyle = 'dashed')
    plt.plot(copies, np.mean(score_fidn, 1),  color='red', label='Fidelity negative', linestyle = 'dashdot')
    plt.xlabel('copies')
    plt.ylabel('Classification score')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"classification scores of {dataset}")
    plt.savefig(f'{figures_path}/score_figures/{dataset}_score_final.png')
    plt.savefig(f'{figures_path}/score_figures/{dataset}_score_final.pdf')
    plt.show()


def plot_3d(dataset):

    score_p = np.load(f'{output_path}/{dataset}_cross_output/{dataset}_score_p.npy')

    score_n = np.load(f'{output_path}/{dataset}_cross_output/{dataset}_score_n.npy')

    score_fidp = np.load(f'{output_path}/{dataset}_cross_output/{dataset}_score_fidp.npy')

    score_fidn = np.load(f'{output_path}/{dataset}_cross_output/{dataset}_score_fidn.npy')


    fig: Figure = plt.figure(figsize=(8, 8))
    ax_mw: Axes = fig.add_subplot(projection='3d')

    x = [3,99,199,299,399]
    for i in x:
        hist, bins = np.histogram(score_p[i], bins=30)
        hist1, bins1 = np.histogram(score_n[i], bins = 30)
        ax_mw.bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='blue', alpha=0.7, zdir='x', zs=(i+1)/4)
        ax_mw.bar(bins1[:-1], hist1, width=(bins1[1] - bins1[0]), color='red', alpha=0.7, zdir='x', zs=(i+1)/4)

    ax_mw.set_xlabel('No. of copies', fontsize=15)
    ax_mw.set_ylabel('Distribution of classification scores', fontsize=15)
    ax_mw.set_zlabel('Histogram', fontsize=15)

    ax_mw.view_init(30, 235)
    plt.tight_layout()
    plt.savefig(f'{figures_path}/{dataset}_cross_figures/{dataset}_3d_helstrom.png')
    
    plt.show()
    
    
    fig: Figure = plt.figure(figsize=(8, 8))
    ax_mw: Axes = fig.add_subplot(projection='3d')

    x = [3,99,199,299,399]
    for i in x:
        hist, bins = np.histogram(score_fidp[i], bins=30)
        hist1, bins1 = np.histogram(score_fidn[i], bins = 30)
        ax_mw.bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='darkblue', alpha=0.7, zdir='x', zs=(i+1)/4)
        ax_mw.bar(bins1[:-1], hist1, width=(bins1[1] - bins1[0]), color='darkred', alpha=0.7, zdir='x', zs=(i+1)/4)

    ax_mw.set_xlabel('No. of copies', fontsize=15)
    ax_mw.set_ylabel('Distribution of classification scores', fontsize=15)
    ax_mw.set_zlabel('Histogram', fontsize=15)

    ax_mw.view_init(30, 235)
    plt.tight_layout()
    plt.savefig(f'{figures_path}/{dataset}_cross_figures/{dataset}_3d_fidelity.png')
    
    plt.show()


def avg_score_plot(dataset):
    if dataset == 'wine':
        df = pd.read_csv(f'{output_path}/{dataset}_cross_output/{dataset}_cross_data_1_300.csv')
    else:
        df = pd.read_csv(f'{output_path}/{dataset}_cross_output/{dataset}_cross_data_1_100.csv')
    
    fig: Figure = plt.figure(figsize =(10, 8))
    ax: Axes = fig.subplots()
    
    ax.plot(df['copies'], df['score_p'], color='tab:blue', label= 'Helstrom Simulation')
    ax.plot(df['copies'], df['score_n'], color='tab:blue', linestyle = 'dotted')
    ax.plot(df['copies'], df['score_fidp'], color='tab:red', label= 'Fidelity Simulation', linestyle = 'dashed')
    ax.plot(df['copies'], df['score_fidn'], color='tab:red', linestyle = 'dashdot')
    ax.axhline(y=0, color='black', linewidth = 0.8)
    ax.tick_params(axis='y', labelcolor='black')
    
    ax.set_xlabel('Copies', color='black')
    ax.set_ylabel('Classification score', color='black')
    plt.legend()
    plt.savefig(f'{figures_path}/avg_score/{dataset}_score.png')
    plt.show()
