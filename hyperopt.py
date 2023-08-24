from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import time


param_hyperopt = {
    'loss': hp.choice('loss', ['ls', 'lad', 'huber', 'quantile']),
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'min_child_samples': scope.int(hp.quniform('min_samples_leaf', 20, 800, 20)),
    'subsample': scope.float(hp.quniform('subsample', 0.5, 1.0, 0.05)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 5, 1)),
    "num_leaves": scope.int(hp.quniform('num_leaves', 10, 70, 5)),
    #'class_weight': {0:1, 1:scope.int(hp.quniform('class_weight', 1, 27, 1))},
    "min_split_gain":scope.int(hp.quniform('min_split_gain', 1, 15, 1)), 
    "reg_alpha":scope.int(hp.quniform('reg_alpha', 0, 100, 1)),
    "reg_lambda":scope.int(hp.quniform('reg_lambda', 0, 100, 1)),
    "colsample_bytree": scope.float(hp.quniform('feature_fraction', 0.1, 0.9, 0.05))
}

def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):
    '''
    Bayesian hyperparameter optimalization.
    
    Parameters
    ----------
    param_space : the list of hyperparameters and the potencial ranges/options to choose from
    
    X_train, y_train, X_test, y_test : dataframes
    
    num_eval : number of evaluation steps 
    
    Returns
    -------
    best_param : dictionary with the best parameter combination
    '''
    start = time.time()
    
    def objective_function(params):
        gbr = lgb.LGBMClassifier(**params, random_state = 42)
        np.random.seed(42)
        score = cross_val_score(gbr, X_train, y_train,scoring='neg_mean_squared_error', cv=4).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.default_rng(42)
                     )
    
    return best_param

num_eval = 100
best_params = hyperopt(param_hyperopt, X_train, y_train, X_test, y_test, num_eval)
