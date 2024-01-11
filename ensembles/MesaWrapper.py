from .MESA.mesa import Mesa
from sklearn.preprocessing import binarize
import argparse
import numpy as np

class MesaWrapper(Mesa):
    def __init__(self, estimator, n_estimators=10, random_state=None, **kwargs):

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
        args.update_steps = 1000
        args.start_steps = 500
        args.sigma = 0.2
        args.max_estimators = 10
        args.meta_verbose = 10
        args.meta_verbose_mean_episodes = 25
        args.verbose = False

        super().__init__(args=args, base_estimator=estimator, n_estimators=n_estimators, **kwargs)

    def predict(self, X, **kwargs):
        if "th" in kwargs:
            threshold = kwargs["th"]
        else:
            threshold = 0.5
        y_pred_binarized = binarize(
            self.predict_proba(X)[:,1].reshape(1,-1), threshold=threshold)[0]
        return y_pred_binarized