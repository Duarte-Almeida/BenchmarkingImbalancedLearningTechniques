from imblearn.under_sampling import RandomUnderSampler

class RUSWrapper(RandomUnderSampler):

    def __init__(self, categorical_features = 0, random_state = 42):
        self.categorical_features = categorical_features
        self.random_state = random_state
        super().__init__()

    def fit_resample(self, X, y, *args, **kwargs):
        return super().fit_resample(X, y)
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def parameter_grid(self):
        return {
            'sampling_strategy': uniform(0, 1),
        }

