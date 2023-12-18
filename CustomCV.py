import numpy as np
from sklearn.model_selection import ParameterGrid, PredefinedSplit, RandomizedSearchCV
from itertools import product
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

def get_train_test_splits(X, y, prep, is_oversampler, is_undersampler):
    print("Generating folds")
    n_splits = 2
    skf = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 42)
    X_final = X.copy()
    y_final = y.copy()
    splits = []
    num_resampled = 0
    counter = X.shape[0]
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if is_oversampler and is_undersampler:
            X_aux, y_aux = X[train_index, :], y[train_index]
            X_tr, y_tr = prep.fit_resample(X_aux, y_aux)
            aux_idx = self.resampled_idx_
            X_r, y_r = X_tr[aux_idx:], y_tr[aux_idx:]
            train_index = self.sample_indices_
            train_index = np.concatenate((train_index, np.arange(counter, counter + X_r.shape[0])))
            counter += X_r.shape[0]
            splits.append((train_index, test_index))
            X_final = np.concatenate((X_final, X_r), axis = 0)
            y_final = np.concatenate((y_final, y_r), axis = 0)

        if is_oversampler:
            X_aux, y_aux = X[train_index, :], y[train_index]
            try:
                X_tr, y_tr = prep.fit_resample(X_aux, y_aux)
            except RuntimeError:
                raise RuntimeError
            X_r, y_r = X_tr[X_aux.shape[0]:], y_tr[y_aux.shape[0]:]
            train_index = np.concatenate((train_index, np.arange(counter, counter + X_r.shape[0])))
            counter += X_r.shape[0]
            splits.append((train_index, test_index))
            X_final = np.concatenate((X_final, X_r), axis = 0)
            y_final = np.concatenate((y_final, y_r), axis = 0)

        if is_undersampler:
            X_aux, y_aux = X[train_index, :], y[train_index]
            X_tr, y_tr = prep.fit_resample(X_aux, y_aux)
            train_index = prep.sample_indices_
            splits.append((train_index, test_index))

    return X_final, y_final, splits

class CustomCV(RandomizedSearchCV):
    '''
    Custom GridSearchCV wrapper that freezes the preprocessor configuration (e.g., oversampler)
    and then tries out parameter combinations of downstream components to speed up cross validation
    '''
    def __init__(self, estimator, oversampler, undersampler, prep_name = None, *args, **kwargs):
        super().__init__(estimator = estimator, *args, **kwargs)
        self.oversampler = oversampler
        self.undersampler = undersampler
        # If a preprocessor exists, remove it from the pipeline and unpack
        # the parameters in its parameter grid 
        self.estimator = estimator
        self.prep_name = prep_name
        self.random_state = 42
        if self.prep_name is not None:
            self.prep = self.estimator.named_steps[prep_name]

            self.estimator.steps.pop(0)
            prep_params = {key[len(prep_name) + 2:]: value for key, value in self.param_distributions.items() if key.startswith(prep_name)}
            self.param_distributions = {key: value for key, value in self.param_distributions.items() if not key.startswith(prep_name)}

            if prep_params != {}:
                keys, values = zip(*prep_params.items())
                self.param_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
            else:
                self.param_combinations = []

    def fit(self, X, y, *args, **kwargs):

        best_params_ = []
        best_score_ = -np.inf
        prep_params = None
        best_estimator_ = None
        best_prep_params_ = None

        X = X.to_numpy()
        y = y.to_numpy()

        if self.prep_name is None:
            self.refit = True
            super().fit(X, y, *args, **kwargs)
            return self.best_estimator_
        
        if self.param_combinations == []:
            X_tr, y_tr, splits = get_train_test_splits(X, y, self.prep, self.oversampler, self.undersampler)
            setattr(self, "cv", splits)
            self.refit = True
            super().fit(X_tr, y_tr, *args, **kwargs)
            print("Best parameter (CV score=%0.3f):" %  self.best_score_)
            print(self.best_params_)
            return self.best_estimator_

        # Perform the two-step grid search
        for params in self.param_combinations:
            self.prep.set_params(**params)
            try:
                X_tr, y_tr, splits = get_train_test_splits(X, y, self.prep, self.oversampler, self.undersampler)
            except RuntimeError:
                continue

            setattr(self, "cv", splits)
            super().fit(X_tr, y_tr, *args, **kwargs)


            if self.best_score_ > best_score_:
                best_score_ = self.best_score_
                best_params_ = self.best_params_
                best_prep_params_ = params


        # refit the estimator
        self.prep.set_params(**best_prep_params_)
        X_tr, y_tr = self.prep.fit_resample(X, y)
        self.estimator.set_params(**best_params_)
        self.estimator = self.estimator.fit(X_tr, y_tr, *args, **kwargs)

        for (param, value) in best_prep_params_.items():
            best_params_[self.prep_name + "__" + param] = value
        
        
        self.best_params_ = best_params_
        self.best_score_ = best_score_
        self.best_estimator_ = self.estimator

        self.refit = True
                
        print("Best parameter (CV score=%0.3f):" %  self.best_score_)
        print(self.best_params_)
        return self.best_estimator_

