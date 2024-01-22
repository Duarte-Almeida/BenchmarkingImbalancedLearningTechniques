import utils
from functools import *
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import optuna
import classifiers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

np.random.seed(42)
dataset = "baf"

data = utils.fetch_data(dataset)
X_train, y_train = data["train"]
X_test, y_test = data["test"]
cat_feats = data["cat_feats"]

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[:, -len(cat_feats):])
X_train_non_cat = X_train[:, :-len(cat_feats)]
X_train = np.concatenate((X_train_non_cat, X_train_cat), axis=1)

X_test_cat = encoder.fit_transform(X_test[:, -len(cat_feats):])
X_test_non_cat = X_test[:, :-len(cat_feats)]
X_test = np.concatenate((X_test_non_cat, X_test_cat), axis=1)

model = classifiers.NeuralNetwork()
params = model.parameter_grid()

# hyperparameter search based on ROC-AUC
def objective(trial, X_aux, y_aux, param_grid, cv_object, estimator):
    params = {param: getattr(trial, value[0])(param, *value[1:]) for (param, value) in param_grid.items()}
    scores = []

    for i, (train_index, test_index) in enumerate(cv_object):

        X_train, y_train = X_aux[train_index], y_aux[train_index]
        X_test, y_test = X_aux[test_index], y_aux[test_index]
        
        estimator.set_params(**params)
        estimator.fit(X_train, y_train, epochs = 3)

        probs = estimator.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(y_test, probs))
        print(f"Score obtained on {i}-th fold: {scores[-1]}")

    return np.mean(scores)

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
cv = [idx for idx in skf.split(X_train, y_train)]
clf_objective = partial(objective, X_aux = X_train, y_aux = y_train, param_grid = params,
                                cv_object = cv, estimator = model)

# get best classifier given previously obtained best estimator
print(f"Finding best hyperparameter combinations...")

study = optuna.create_study(direction='maximize', sampler = optuna.samplers.TPESampler(seed = 42))
study.optimize(clf_objective, n_trials = 20)

best_score_ = study.best_value
best_params_ = study.best_params


