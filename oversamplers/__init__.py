from .SMOTE import SMOTEWrapper
from .ADASYN import ADASYNWrapper
from .KMeansSMOTE import KMeansSMOTEWrapper
from .RACOG import RACOGWrapper

oversampler_names = {
    'SMOTE':SMOTEWrapper, 
    'ADASYN':ADASYNWrapper,
    'KMeansSMOTE':KMeansSMOTEWrapper,
    'RACOG': RACOGWrapper
}

def fetch_oversampler(name):
    assert name in oversampler_names
    return oversampler_names[name]()