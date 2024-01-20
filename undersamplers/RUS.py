from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import uniform
from skopt import space
import numpy as np
import pandas as pd
import sys

class RUSWrapper(RandomUnderSampler):

    def __init__(self, categorical_features = 0, cls = "majority", random_state = 42):
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.cls = cls
        self.sampling_ratio = 1
        super().__init__()

    def fit_resample(self, X, y, *args, **kwargs):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        neg = y[y == 0].shape[0]
        pos = y[y == 1].shape[0]
        IR = pos / neg
        eps = 1 / pos

        rng = np.random.default_rng(self.random_state)   
        print(f"Sampling strategy: {self.sampling_ratio}")

        if self.cls == "majority":
            self.sampling_strategy = min(1, IR + self.sampling_ratio * (1 - IR) + eps)
            #print(f"My sampling strategy is {self.sampling_strategy}")
            num_samples = int(pos / self.sampling_strategy)
            print(f"Will only use {num_samples} out of {neg}")
            neg_idx = np.where(y == 0)[0]
            pos_idx = np.where(y == 1)[0]
            idx = rng.choice(neg_idx, size = num_samples, replace = False)
            idx = np.concatenate((idx, pos_idx))
            self.sample_indices_ = np.sort(idx)
            return X[idx], y[idx]
            
        else:
            self.sampling_strategy = min(1, (1 - self.sampling_ratio) + eps)
            num_samples_pos = int(pos * self.sampling_strategy)
            num_samples_neg = int(neg * self.sampling_strategy)

            neg_idx = np.where(y == 0)[0]
            pos_idx = np.where(y == 1)[0]

            pos_idx = rng.choice(pos, size = num_samples_pos, replace = False)
            neg_idx = rng.choice(neg, size = num_samples_neg, replace = False)
            idx = np.concatenate((neg_idx, pos_idx))
            self.sample_indices_ = idx
            return X[idx], y[idx]


    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def parameter_grid(self):
        return {
            'sampling_ratio': ("suggest_uniform", 0.0, 0.1)
        }
    def adapt_hyperparameters(self, X, y):
        pass

