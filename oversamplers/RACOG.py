from .racog import RACOG
from skopt.space import Real, Integer
from skopt.space import Dimension


class RACOGWrapper(RACOG):

    def _fit_resample(self, X, y):
        self.fit(X, y)
        neg = y[y == 0].shape[0]
        pos = y[y == 1].shape[0]
        IR = pos / neg
        eps = 1 / pos

        self.sampling_strategy = min(1, IR + self.sampling_ratio * (1 - IR) + eps)
        return self.sample(X, y)
    def parameter_grid(self):
        return {
            'sampling_ratio': ("suggest_uniform", 0.0, 0.1),
            'offset': ("suggest_categorical", [i for i in range(10, 100)]),
            'lag0': ("suggest_categorical", [i for i in range(5, 10)])
        }
    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._get_param_names()}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def adapt_hyperparameters(self, X, y):
        pass