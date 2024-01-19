from sklearn.model_selection import cross_val_score
import numpy as np

def RF(X, y, mode, hyperparameters):
    def train_RF(hyperparameters):
            # Create and train your RandomForestClassifier
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=hyperparameters['n_estimators'],
                max_depth=hyperparameters['max_depth'],
                min_samples_split=hyperparameters['min_samples_split']
            )
            
            # Evaluate the model's performance
            score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
            return score
    if mode == 'optimize':
        return train_RF
    
    elif mode == 'final':
        return train_RF(hyperparameters)

def LogisticR(X, y, mode, hyperparameters):
    def logisticR(hyperparameters):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=hyperparameters['C'],
            solver=hyperparameters['solver'],
            max_iter=hyperparameters['max_iter'])
        
        # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    if mode == 'optimize':
        return logisticR
        
    elif mode == 'final':
        return logisticR(hyperparameters)


def LDA(X, y, mode, hyperparameters):
    def train_LDA(hyperparameters):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis(solver=hyperparameters['solver'],
            shrinkage=hyperparameters['shrinkage'])
         # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    
    if mode == 'optimize':
        return train_LDA
        
    elif mode == 'final':
        return train_LDA(hyperparameters)
    
def QDA(X, y, mode, hyperparameters):
    def train_QDA(hyperparameters):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis(reg_param=hyperparameters['reg_param'])
         # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    
    if mode == 'optimize':
        return train_QDA
        
    elif mode == 'final':
        return train_QDA(hyperparameters)

def BNB(X, y, mode, hyperparameters):
    def train_BernoulliNB(hyperparameters):
        from sklearn.naive_bayes import BernoulliNB
        model = BernoulliNB(alpha=hyperparameters['alpha'], 
                            binarize=hyperparameters['binarize'])
         # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    if mode == 'optimize':
        return train_BernoulliNB
        
    elif mode == 'final':
        return train_BernoulliNB(hyperparameters)

def SVM(X, y, mode, hyperparameters):
    def train_SVM(hyperparameters):
        from sklearn.svm import SVC
        model = SVC(kernel=hyperparameters['kernel'], 
                    C=hyperparameters['C'],
                    gamma=hyperparameters['gamma'])
         # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    
    if mode == 'optimize':
        return train_SVM
        
    elif mode == 'final':
        return train_SVM(hyperparameters)

def ADA(X, y, mode, hyperparameters):
    def train_ada(hyperparameters):
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=hyperparameters['n_estimators'], 
                    learning_rate=hyperparameters['learning_rate'])
         # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score

    if mode == 'optimize':
        return train_ada
        
    elif mode == 'final':
        return train_ada(hyperparameters)

def KNN(X, y, mode, hyperparameters):
    def train_KNN(hyperparameters):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(
                    weights=hyperparameters['weights'], 
                    n_neighbors=hyperparameters['n_neighbours'],
                    metric=hyperparameters['metrics']
                    )
         # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    if mode == 'optimize':
        return train_KNN
        
    elif mode == 'final':
        return train_KNN(hyperparameters)

def DT(X, y, mode, hyperparameters):
    def train_DT(hyperparameters):
        from sklearn import tree
        model = tree.DecisionTreeClassifier(criterion=hyperparameters['criterion'], 
                    max_depth=hyperparameters['max_depth'],
                    min_samples_split=hyperparameters['min_samples_split'])
         # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    if mode == 'optimize':
        return train_DT
        
    elif mode == 'final':
        return train_DT(hyperparameters)
    
def NearestC(X, y, mode, hyperparameters):
    def train_NC(hyperparameters):
        from sklearn.neighbors import NearestCentroid
        model = NearestCentroid(metric=hyperparameters['metric'], 
                    shrink_threshold=hyperparameters['shrink_threshold'])
         # Evaluate the model's performance
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    if mode == 'optimize':
        return train_NC
        
    elif mode == 'final':
        return train_NC(hyperparameters)
    
def XG(X, y, mode, hyperparameters):
    def train_XG(hyperparameters):
        import xgboost as xgb
        model = xgb.XGBClassifier(
            learning_rate=hyperparameters['learning_rate'],
            verbosity=hyperparameters['verbosity'],
            n_estimators=180,
            nthread=4,
            booster=hyperparameters['booster']
            # gamma=hyperparameters['gamma'],
            # reg_alpha=hyperparameters['reg_alpha'],
            # min_child_weight=hyperparameters['min_child_weight']
            )
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    
    if mode == 'optimize':
        return train_XG
        
    elif mode == 'final':
        return train_XG(hyperparameters)
    
def Cat(X, y, mode, hyperparameters):
    def train_Cat(hyperparameters):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            learning_rate=hyperparameters['learning_rate'],
            iterations=hyperparameters['iterations'],
            depth=hyperparameters['depth']
            #l2_leaf_reg=hyperparameters['l2_leaf_reg']
            )
        score = np.mean(cross_val_score(estimator=model, X=X,y=y, cv=5, scoring="f1"))
        return score
    
    if mode == 'optimize':
        return train_Cat
        
    elif mode == 'final':
        return train_Cat(hyperparameters)