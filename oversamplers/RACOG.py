from .racog import RACOG

class RACOGWrapper(RACOG):

        def _fit_resample(self, X, y):
            self.fit(X, y)
            return self.sample(X, y)
        def parameter_grid(self):
            return {
                'sampling_strategy': [0.1, 0.2, 0.25, 0.5, 0.75, 1.00]
            }
        def get_params(self, deep=True):
            return {param: getattr(self, param)
                    for param in self._get_param_names()}
        
        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self
