import numpy as np
from sklearn.model_selection import ParameterGrid, PredefinedSplit, RandomizedSearchCV
from itertools import product
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline

import sys


class CustomCV():
    '''
    Custom GridSearchCV wrapper that freezes the preprocessor configuration (e.g., oversampler)
    and then tries out parameter combinations of downstream components to speed up cross validation
    '''

    def __init__(self, estimator, param_distributions, n_iter, prep_param_distributions = {}, prep_n_iter = 0, *args, **kwargs):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.prep_param_distributions = prep_param_distributions

        self.search = RandomizedSearchCV(estimator = estimator, param_distributions = param_distributions, 
                                    n_iter = n_iter, refit = False, *args, **kwargs)
        if self.prep_param_distributions != {}:
            self.prep_search = RandomizedSearchCV(estimator = estimator, param_distributions = prep_param_distributions, 
                                            n_iter = prep_n_iter, refit = False, *args, **kwargs)
      
    def fit(self, X, y, *args, **kwargs):

        # two step randomized cross validation
        # get best preprocessor parameters given fixed custom classifier
        if self.prep_param_distributions != {}:
            print(f"Finding best preprocessor...")
            self.prep_search.fit(X, y, *args, **kwargs)
            for (param, value) in self.prep_search.best_params_.items():
                self.param_distributions.update({param:[value]})
                    
            #self.search.estimator.set_params(**self.prep_search.best_params_)
            print(f"Best preprocessor config: {self.prep_search.best_params_}")
            print(f"with score: {self.prep_search.best_score_}")

        # get best classifier given previously obtained best estimator
        print(f"Finding best classifier...")
        self.search.fit(X, y, *args, **kwargs)
        print("Best parameter (CV score=%0.3f):" %  self.search.best_score_)
        print(f"{self.search.best_params_}")

        self.search.best_estimator_ = self.search.estimator
        for (attr, value) in self.search.best_params_.items():
            idx = attr.find("__")
            component = attr[:idx]
            attr_name = attr[idx + 2:]
            self.search.best_estimator_.named_steps[component].set_params(**{attr_name: value})
        self.search.best_estimator_.fit(X, y)
        self.best_params_ = self.search.best_params_
        return self.search.best_estimator_