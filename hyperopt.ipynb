from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.model_selection import cross_val_score


import time
from sklearn.model_selection import GridSearchCV

param_hyperopt = {
    'loss': hp.choice('loss', ['ls', 'lad', 'huber', 'quantile']),
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'min_child_samples': scope.int(hp.quniform('min_samples_leaf', 20, 800, 20)),
    'subsample': scope.float(hp.quniform('subsample', 0.5, 1.0, 0.05)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
    "num_leaves": scope.int(hp.quniform('num_leaves', 10, 70, 5)),
    'class_weight': {0:1, 1:scope.int(hp.quniform('class_weight', 1, 27, 1))},
    "min_split_gain":scope.int(hp.quniform('min_split_gain', 1, 15, 1)), 
    "reg_alpha":scope.int(hp.quniform('reg_alpha', 0, 100, 1)),
    "reg_lambda":scope.int(hp.quniform('reg_lambda', 0, 100, 1)),
    "colsample_bytree": scope.float(hp.quniform('feature_fraction', 0.1, 0.9, 0.05)),
#    "bagging_fraction": scope.float(hp.quniform('bagging_fraction', 0.1, 0.9, 0.05)),
   # "early_stopping_rounds": scope.int(hp.quniform('early_stopping_rounds', 10, 100, 10))
}


def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):
    
    start = time.time()
    
    def objective_function(params):
        gbr = LGBMClassifier(**params, random_state = 42)#fsele(**params, random_state = 42)
        np.random.seed(42)
        score = cross_val_score(gbr, X_train, y_train,scoring='roc_auc', cv=4).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.default_rng(42)
                     )
    print(best_param)

num_eval = 1000
best_params, results_hyperopt = hyperopt(param_hyperopt, X_train, y_train, X_test, y_test, num_eval)
