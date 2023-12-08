from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN, KMeansSMOTE, BorderlineSMOTE
from imblearn import FunctionSampler
from imblearn.base import SamplerMixin
from sklearn.utils import check_X_y

from smote_variants import A_SUWO

class SMOTENCWrapper(SMOTENC):

    def __init__(self):
        super(SMOTENCWrapper, self).__init__(categorical_features = "auto")
    
    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._get_param_names()}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def parameter_grid(self):
        return {
            'k_neighbors':[3, 5, 7], 
            'sampling_strategy':[0.1, 0.2, 0.5]
        }

class SMOTEWrapper(SMOTE):

    def parameter_grid(self):
        return {
            'k_neighbors':[3, 5, 7], 
            'sampling_strategy':[0.1, 0.2, 0.5]
        }


class ADASYNWrapper(ADASYN):

    def __init__(self):
        super(ADASYNWrapper, self).__init__()

    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._get_param_names()}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def parameter_grid(self):
        return {
            'n_neighbors':[3, 5, 7], 
            'sampling_strategy':[0.1, 0.2, 0.5]
        }


class KMeansSMOTEWrapper(KMeansSMOTE):

    def parameter_grid(self):
        return {
            'k_neighbors':[3, 5, 7], 
            'sampling_strategy':[0.1, 0.2, 0.5],  
            'cluster_balance_threshold': [0.01]
        }


oversampler_names = {
    'SMOTENC':SMOTENCWrapper,
    'SMOTE':SMOTEWrapper, 
    'ADASYN':ADASYNWrapper, 
    'KMeansSMOTE':KMeansSMOTEWrapper
}

def fetch_oversampler(name):
    assert name in oversampler_names
    return oversampler_names[name]()
