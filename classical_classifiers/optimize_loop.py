from sklearn.model_selection import cross_val_score
from ax.service.managed_loop import optimize
import numpy as np
import train_model
import pandas as pd
import logging
import sys
import os
sys.path.append("./")
from classical_classifiers import output_path
from helstrom_classifier.load_data import load_datasets

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

def  optimize_loop(parameters, evaluation_function):
    best_parameters, best_values, experiemt, model = optimize(
        parameters=parameters,
        evaluation_function=evaluation_function,
        objective_name='f1',
        total_trials=20  # Number of optimization iterations
        )
    #storing the optimal results and hyperparameters
    means, covariances = best_values
    return best_parameters, means

def optimize_parameters(parameters, classifiers, dataset):
    
    X, y = load_datasets(dataset)
    f1 = np.zeros((len(classifiers)))    

    best_hyperparameters = {}

    for i in range(len(classifiers)):
            if classifiers[i] == "LogisticR_lbfgs":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
            
            elif classifiers[i] == "LogisticR_liblinear":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == "LogisticR_newtonc":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                
            
            elif classifiers[i] == "LogisticR_newtonch":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == "LogisticR_sag":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
            
            elif classifiers[i] == "LogisticR_saga":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
    

            
            elif classifiers[i] == "LDA_le":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LDA(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
            
            elif classifiers[i] == "QDA":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.QDA(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == "BernoulliNB":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.BNB(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                
            elif classifiers[i] == "SVM":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.SVM(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == "AdaBoost":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.ADA(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == "RandomForest":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.RF(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == "KNN":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.KNN(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == "DecisionTree":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.DT(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == 'NearestC':
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.NearestC(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters

            elif classifiers[i] == 'GaussianNB':
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB()
                score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
                f1[i] = score
            
            elif classifiers[i] == "XGBooster":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.XG(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
            
            elif classifiers[i] == "Catboost":
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.Cat(X, y, 'optimize', parameters[classifiers[i]]))
                f1[i] = means['f1']
                
            else:
                LOG.info('the classifier does not exist!')

    d = {'clssifiers': classifiers, 'f1_score': f1}
    df = pd.DataFrame(data=d)
    
    path = f'{output_path}/classical_output'
    if not os.path.exists(path):
        os.mkdir(path)
    df.to_csv(f'{output_path}/classical_output/f1_{dataset}.csv')
    return df