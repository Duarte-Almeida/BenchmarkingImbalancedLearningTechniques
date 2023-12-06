from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, BorderlineSMOTE
from imblearn import FunctionSampler
from smote_variants import A_SUWO

class SMOTEWrapper(SMOTE):

    def parameter_grid(self):
        return {
            'k_neighbors':[3, 5, 7], 
            'sampling_strategy':[0.1, 0.2, 0.5]
        }


class ADASYNWrapper(ADASYN):

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
    'SMOTE':SMOTEWrapper, 
    'ADASYN':ADASYNWrapper, 
    'KMeansSMOTE':KMeansSMOTEWrapper
}

def fetch_oversampler(name):
    assert name in oversampler_names
    return oversampler_names[name]()
