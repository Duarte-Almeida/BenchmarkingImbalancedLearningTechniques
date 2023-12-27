from .racog import RACOG
from scipy.stats import uniform, loguniform, rv_discrete

class RACOGWrapper(RACOG):

        def _fit_resample(self, X, y):
            self.fit(X, y)
            return self.sample(X, y)
        def parameter_grid(self):
            values = np.arange(50, 500)
            probabilities = loguniform.cdf(values + 1, 5, 501) - loguniform.cdf(values, 5, 501)
            probabilities = probabilities / np.sum(probabilities)
            return {
                'sampling_strategy':uniform(0, 1),
                'offset': randint(5, 20),
                'lag0': rv_discrete(values = (values, probabilities))
            }
        def get_params(self, deep=True):
            return {param: getattr(self, param)
                    for param in self._get_param_names()}
        
        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self