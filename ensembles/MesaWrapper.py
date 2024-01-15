from .MESA.mesa import Mesa
from sklearn.preprocessing import binarize
import argparse
import numpy as np
import classifiers
from sklearn.model_selection import train_test_split

class MesaWrapper(Mesa):
    def __init__(self, base_estimator = classifiers.LGBM(), n_estimators=10, random_state=42, **kwargs):

        # See MESA/arguments.py for more information
        args = argparse.Namespace()
        args.metric = 'aucprc'
        args.num_bins = 5
        args.gamma = 0.99
        args.tau = 0.01
        args.alpha = 0.1
        args.lr = 0.001
        args.policy = 'Gaussian'
        args.target_update_interval = 1
        args.automatic_entropy_tuning = False
        args.cuda = False
        args.hidden_size = 50
        args.lr_decay_steps = 10
        args.lr_decay_gamma = 0.99
        args.replay_size = 1000
        args.random_state = random_state
        args.seed = random_state
        args.train_ratio = 1
        args.train_ir = 1
        args.reward_coefficient = 100
        args.update_steps = 2
        args.start_steps = 1
        args.sigma = 0.2
        args.max_estimators = 10
        args.meta_verbose = 10
        args.meta_verbose_mean_episodes = 25
        args.verbose = False
        args.updates_per_step = 1
        args.batch_size = 64

        self.grid = {
            #'lr': ('suggest_loguniform', 1e-4, 1e-1),
            #'gamma': ('suggest_uniform', 0.9, 0.999),
            #'tau': ('suggest_uniform', 0.001, 0.1),
            #'batch_size': ('suggest_int', 32, 128),
            #'hidden_size': ('suggest_int', 20, 100),
            #'updates_per_step': ('suggest_int', 1, 5),
            #'target_update_interval': ('suggest_int', 1, 10),
            #'replay_size': ('suggest_int', 500, 5000),
            #'reward_coefficient': ('suggest_uniform', 50, 150),
            #'num_bins': ('suggest_int', 5, 100),
            #'sigma': ('suggest_uniform', 0.1, 0.5),
            'num_estimators': ('suggest_int', 5, 20),
            #'max_estimators': ('suggest_int', 5, 20),
            #'train_ir': ('suggest_uniform', 0.5, 1.5),
            #'train_ratio': ('suggest_uniform', 0.8, 1.0)
        }

        self.param_names = [key for key in self.grid.keys()]

        super().__init__(args=args, base_estimator=base_estimator, n_estimators=n_estimators, **kwargs)

    def fit(self, X, y):
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.20, random_state=42)

        print(f"Training the meta sampler...")
        self.meta_fit(X_train, y_train, X_dev, y_dev)

        print(f"Training the ensemble...")
        super().fit(X_train, y_train, X_dev, y_dev)

        return self

    def predict(self, X, **kwargs):
        if "th" in kwargs:
            threshold = kwargs["th"]
        else:
            threshold = 0.5
        y_pred_binarized = binarize(
            self.predict_proba(X)[:,1].reshape(1,-1), threshold=threshold)[0]
        return y_pred_binarized

    def get_params(self, deep=True):
        res = {}
        other_args = vars(self.env.args)
        for param in self.param_names:
            if hasattr(self, param):
                res[param] = getattr(self, param)
            elif param in other_args:
                res[param] = other_args[param]
        self.env.args = argparse.Namespace(**other_args)
        return res
    
    def set_params(self, **params):
        print(f"Received {params}")
        if not params:
            return self
        attr_params = {}
        other_args = vars(self.env.args)
        for key, value in params.items():
            print(f"Trying to set {key} to {value}")
            if '__' in key:
                print(1)
                idx = key.find('__')
                attr = key[:idx]
                attr_param = key[idx + 2:]
                if hasattr(self, attr):
                    if attr not in attr_params:
                        attr_params[attr] = {}
                    attr_params[attr][attr_param] = value
            if hasattr(self, key):
                print(2)
                if key == "base_estimator":
                    setattr(self.env, key, value)
                setattr(self, key, value)
            elif key in other_args:
                print(3)
                #setattr(self.env, "args", value)
                other_args[key] = value
            else:
                print(4)
                #print(self.env.args)
        self.env.args = argparse.Namespace(**other_args)
        print(self.env.args)


        return self

    def parameter_grid(self):
        return self.grid
    
    def adapt_hyperparameters(self, X, y):
        self.base_estimator.adapt_hyperparameters(X, y)

    def _get_param_names(self, X, y):
        return []